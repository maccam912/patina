"""Consolidation job scheduler using APScheduler."""

from datetime import datetime
from typing import Optional, List, Callable, Any
from uuid import UUID
import logging

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False
    AsyncIOScheduler = None
    CronTrigger = None

from patina.database.connection import Database
from patina.consolidation.consolidator import MemoryConsolidator

logger = logging.getLogger(__name__)


class ConsolidationScheduler:
    """Scheduler for memory consolidation jobs.
    
    Manages:
    - Daily consolidation (journal generation) - runs at configured hour
    - Weekly synthesis - runs Sundays
    - Monthly integration - runs first of month
    - Memory decay cycle - runs hourly
    """
    
    def __init__(
        self,
        db: Database,
        consolidator: Optional[MemoryConsolidator] = None,
        llm_client=None,
    ):
        self.db = db
        self.consolidator = consolidator or MemoryConsolidator(db, llm_client)
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._agents: List[dict] = []
    
    async def start(self, agent_ids: Optional[List[UUID]] = None) -> None:
        """Start the consolidation scheduler.
        
        Args:
            agent_ids: Specific agents to schedule (all if None)
        """
        if not HAS_APSCHEDULER:
            logger.warning("APScheduler not installed. Scheduling disabled.")
            return
        
        # Load agents to schedule
        await self._load_agents(agent_ids)
        
        if not self._agents:
            logger.warning("No agents found for scheduling.")
            return
        
        self._scheduler = AsyncIOScheduler()
        
        for agent in self._agents:
            await self._schedule_agent_jobs(agent)
        
        self._scheduler.start()
        logger.info(f"Started consolidation scheduler for {len(self._agents)} agent(s)")
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Stopped consolidation scheduler")
    
    async def _load_agents(self, agent_ids: Optional[List[UUID]] = None) -> None:
        """Load agents that have consolidation enabled."""
        if agent_ids:
            for aid in agent_ids:
                row = await self.db.fetch_one(
                    "SELECT * FROM agents WHERE id = :id AND consolidation_enabled = 1",
                    {"id": str(aid)},
                )
                if row:
                    self._agents.append(row)
        else:
            rows = await self.db.fetch_all(
                "SELECT * FROM agents WHERE consolidation_enabled = 1"
            )
            self._agents = rows
    
    async def _schedule_agent_jobs(self, agent: dict) -> None:
        """Schedule all jobs for a single agent."""
        agent_id = UUID(agent["id"])
        tenant_id = UUID(agent["tenant_id"])
        hour = agent.get("consolidation_hour", 3)  # Default 3 AM
        
        # Daily consolidation
        self._scheduler.add_job(
            self._run_daily,
            CronTrigger(hour=hour, minute=0),
            args=[agent_id, tenant_id],
            id=f"daily_{agent_id}",
            replace_existing=True,
        )
        
        # Weekly synthesis (Sundays at hour+1)
        self._scheduler.add_job(
            self._run_weekly,
            CronTrigger(day_of_week=6, hour=hour + 1, minute=0),
            args=[agent_id, tenant_id],
            id=f"weekly_{agent_id}",
            replace_existing=True,
        )
        
        # Monthly integration (1st of month at hour+2)
        self._scheduler.add_job(
            self._run_monthly,
            CronTrigger(day=1, hour=hour + 2, minute=0),
            args=[agent_id, tenant_id],
            id=f"monthly_{agent_id}",
            replace_existing=True,
        )
        
        # Decay cycle (hourly)
        self._scheduler.add_job(
            self._run_decay,
            CronTrigger(minute=30),  # 30 minutes past each hour
            args=[agent_id, tenant_id],
            id=f"decay_{agent_id}",
            replace_existing=True,
        )
        
        logger.info(f"Scheduled consolidation jobs for agent {agent_id}")
    
    async def _run_daily(self, agent_id: UUID, tenant_id: UUID) -> None:
        """Run daily consolidation."""
        try:
            result = await self.consolidator.run_daily_consolidation(
                agent_id=agent_id,
                tenant_id=tenant_id,
            )
            logger.info(f"Daily consolidation completed for {agent_id}: {result}")
        except Exception as e:
            logger.error(f"Daily consolidation failed for {agent_id}: {e}")
    
    async def _run_weekly(self, agent_id: UUID, tenant_id: UUID) -> None:
        """Run weekly synthesis."""
        try:
            result = await self.consolidator.run_weekly_synthesis(
                agent_id=agent_id,
                tenant_id=tenant_id,
            )
            logger.info(f"Weekly synthesis completed for {agent_id}: {result}")
        except Exception as e:
            logger.error(f"Weekly synthesis failed for {agent_id}: {e}")
    
    async def _run_monthly(self, agent_id: UUID, tenant_id: UUID) -> None:
        """Run monthly integration."""
        try:
            result = await self.consolidator.run_monthly_integration(
                agent_id=agent_id,
                tenant_id=tenant_id,
            )
            logger.info(f"Monthly integration completed for {agent_id}: {result}")
        except Exception as e:
            logger.error(f"Monthly integration failed for {agent_id}: {e}")
    
    async def _run_decay(self, agent_id: UUID, tenant_id: UUID) -> None:
        """Run decay cycle."""
        try:
            result = await self.consolidator.run_decay_cycle(
                tenant_id=tenant_id,
                agent_id=agent_id,
            )
            logger.debug(f"Decay cycle completed for {agent_id}: {result}")
        except Exception as e:
            logger.error(f"Decay cycle failed for {agent_id}: {e}")
    
    async def run_manual(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        job_type: str = "all",
    ) -> dict:
        """Run consolidation jobs manually.
        
        Args:
            agent_id: Agent to run jobs for
            tenant_id: Tenant context
            job_type: One of 'daily', 'weekly', 'monthly', 'decay', or 'all'
            
        Returns:
            Results of the job(s) run
        """
        results = {}
        
        if job_type in ("daily", "all"):
            results["daily"] = await self.consolidator.run_daily_consolidation(
                agent_id=agent_id,
                tenant_id=tenant_id,
            )
        
        if job_type in ("weekly", "all"):
            results["weekly"] = await self.consolidator.run_weekly_synthesis(
                agent_id=agent_id,
                tenant_id=tenant_id,
            )
        
        if job_type in ("monthly", "all"):
            results["monthly"] = await self.consolidator.run_monthly_integration(
                agent_id=agent_id,
                tenant_id=tenant_id,
            )
        
        if job_type in ("decay", "all"):
            results["decay"] = await self.consolidator.run_decay_cycle(
                tenant_id=tenant_id,
                agent_id=agent_id,
            )
        
        return results
