#!/usr/bin/env python3
"""Lightweight watchdog for proxy_gateway and audit_validator.

Periodically pings health endpoints and restarts PM2 processes if unreachable.
Includes a startup grace period so fresh processes have time to load models.
"""

import argparse
import asyncio
import json
import logging
import subprocess
import time

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("watchdog")


class ServiceMonitor:
    def __init__(self, name: str, health_url: str, pm2_name: str,
                 failures_before_restart: int = 3, grace_s: float = 180,
                 deep_check_url: str = "", deep_check_interval: int = 5):
        self.name = name
        self.health_url = health_url
        self.pm2_name = pm2_name
        self.failures_before_restart = failures_before_restart
        self.grace_s = grace_s
        self.consecutive_failures = 0
        self.last_restart_time = time.time()  # grace on startup
        self.total_restarts = 0
        self.deep_check_url = deep_check_url  # e.g. gateway /v1/chat/completions
        self.deep_check_interval = deep_check_interval
        self._check_count = 0

    def in_grace_period(self) -> bool:
        return time.time() - self.last_restart_time < self.grace_s

    async def check(self, session: aiohttp.ClientSession) -> bool:
        # Standard health check
        try:
            async with session.get(self.health_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    if self.consecutive_failures > 0:
                        log.info(f"[watchdog] {self.name}: recovered after {self.consecutive_failures} failures")
                    self.consecutive_failures = 0
                else:
                    log.warning(f"[watchdog] {self.name}: HTTP {resp.status}")
                    self.consecutive_failures += 1
                    log.info(f"[watchdog] {self.name}: failure {self.consecutive_failures}/{self.failures_before_restart}")
                    return False
        except Exception as e:
            log.warning(f"[watchdog] {self.name}: connection failed — {e}")
            self.consecutive_failures += 1
            log.info(f"[watchdog] {self.name}: failure {self.consecutive_failures}/{self.failures_before_restart}")
            return False

        # Deep check: send actual inference request periodically
        self._check_count += 1
        if self.deep_check_url and self._check_count % self.deep_check_interval == 0:
            ok = await self._deep_check(session)
            if not ok:
                self.consecutive_failures += 1
                log.warning(f"[watchdog] {self.name}: deep check FAILED (inference broken despite healthy endpoint)")
                return False

        return True

    async def _deep_check(self, session: aiohttp.ClientSession) -> bool:
        """Send a real inference request to verify end-to-end functionality."""
        payload = {
            "model": "qwen-7b",
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5,
            "stream": False,
        }
        try:
            async with session.post(
                self.deep_check_url, json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    log.warning(f"[watchdog] {self.name}: deep check HTTP {resp.status}")
                    return False
                body = await resp.json()
                choices = body.get("choices", [])
                if not choices or not choices[0].get("message", {}).get("content"):
                    log.warning(f"[watchdog] {self.name}: deep check empty response")
                    return False
                log.info(f"[watchdog] {self.name}: deep check OK")
                return True
        except Exception as e:
            log.warning(f"[watchdog] {self.name}: deep check error — {e}")
            return False

    def should_restart(self) -> bool:
        if self.in_grace_period():
            return False
        return self.consecutive_failures >= self.failures_before_restart

    def restart(self):
        log.warning(f"Restarting {self.name} (attempt #{self.total_restarts + 1})")
        try:
            subprocess.run(
                ["pm2", "restart", self.pm2_name],
                capture_output=True, timeout=30,
            )
            log.info(f"[watchdog] {self.name}: pm2 restart successful")
        except Exception as e:
            log.error(f"[watchdog] {self.name}: restart failed — {e}")
        self.consecutive_failures = 0
        self.last_restart_time = time.time()
        self.total_restarts += 1


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60, help="Check interval seconds")
    parser.add_argument("--gateway-url", default="http://localhost:8081/health")
    parser.add_argument("--auditor-url", default="http://localhost:8082/health")
    parser.add_argument("--proxy-pm2-name", default="proxy_gateway")
    parser.add_argument("--auditor-pm2-name", default="audit_validator")
    parser.add_argument("--grace", type=int, default=180, help="Startup grace period seconds")
    parser.add_argument("--failures", type=int, default=3, help="Failures before restart")
    args = parser.parse_args()

    # Gateway gets a deep check that sends a real inference request every 5th cycle
    gateway_base = args.gateway_url.rsplit("/health", 1)[0]
    monitors = [
        ServiceMonitor("proxy", args.gateway_url, args.proxy_pm2_name,
                        failures_before_restart=args.failures, grace_s=args.grace,
                        deep_check_url=f"{gateway_base}/v1/chat/completions",
                        deep_check_interval=5),
        ServiceMonitor("auditor", args.auditor_url, args.auditor_pm2_name,
                        failures_before_restart=args.failures, grace_s=args.grace),
    ]

    log.info(f"Watchdog started: interval={args.interval}s grace={args.grace}s failures={args.failures}")
    for m in monitors:
        log.info(f"  {m.name}: {m.health_url} -> pm2:{m.pm2_name}")

    async with aiohttp.ClientSession() as session:
        while True:
            for monitor in monitors:
                ok = await monitor.check(session)
                if not ok and monitor.should_restart():
                    monitor.restart()
            await asyncio.sleep(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
