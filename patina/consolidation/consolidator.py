"""Unified consolidator: Orchestrates all consolidation operations."""

from datetime import datetime, date, timedelta
from typing import Dict, Optional, Any
from uuid import UUID

from patina.database.connection import Database
from patina.consolidation.decay import MemoryDecayManager
from patina.consolidation.daily import DailyConsolidator
from patina.consolidation.weekly import WeeklySynthesizer
from patina.consolidation.monthly import MonthlyIntegrator


class MemoryConsolidator:
    """Unified interface for all consolidation operations.
    
    Orchestrates the memory consolidation pipeline:
    - Daily journal generation
    - Weekly synthesis
    - Monthly integration
    - Memory decay management
    """
    
    def __init__(self, db: Database, llm_client=None):
        self.db = db
        self.llm = llm_client
        
        # Initialize consolidators
        self.decay = MemoryDecayManager(db)
        self.daily = DailyConsolidator(db, llm_client)
        self.weekly = WeeklySynthesizer(db, llm_client)
        self.monthly = MonthlyIntegrator(db, llm_client)
    
    async def run_daily_consolidation(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        target_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Run daily consolidation for an agent."""
        target_date = target_date or (datetime.utcnow().date() - timedelta(days=1))
        
        journal = await self.daily.run(
            agent_id=agent_id,
            tenant_id=tenant_id,
            target_date=target_date,
        )
        
        return {
            "type": "daily",
            "date": target_date.isoformat(),
            "journal_created": journal is not None,
            "journal_id": str(journal.id) if journal else None,
        }
    
    async def run_weekly_synthesis(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        week_start: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Run weekly synthesis for an agent."""
        synthesis = await self.weekly.run(
            agent_id=agent_id,
            tenant_id=tenant_id,
            week_start=week_start,
        )
        
        return {
            "type": "weekly",
            "synthesis_created": synthesis is not None,
            "synthesis_id": str(synthesis.id) if synthesis else None,
        }
    
    async def run_monthly_integration(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        target_month: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Run monthly integration for an agent."""
        result = await self.monthly.run(
            agent_id=agent_id,
            tenant_id=tenant_id,
            target_month=target_month,
        )
        
        return {
            "type": "monthly",
            **result,
        }
    
    async def run_decay_cycle(
        self,
        tenant_id: UUID,
        agent_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """Run memory decay cycle."""
        result = await self.decay.run_decay_cycle(
            tenant_id=tenant_id,
            agent_id=agent_id,
        )
        
        return {
            "type": "decay",
            **result,
        }
    
    async def run_full_consolidation(
        self,
        agent_id: UUID,
        tenant_id: UUID,
    ) -> Dict[str, Any]:
        """Run all consolidation stages.
        
        Useful for manual trigger or catch-up consolidation.
        """
        results = {}
        
        # Daily (yesterday)
        results["daily"] = await self.run_daily_consolidation(
            agent_id=agent_id,
            tenant_id=tenant_id,
        )
        
        # Weekly (if it's Sunday or Monday)
        today = datetime.utcnow().date()
        if today.weekday() in [0, 6]:  # Monday or Sunday
            results["weekly"] = await self.run_weekly_synthesis(
                agent_id=agent_id,
                tenant_id=tenant_id,
            )
        
        # Monthly (if first or second day of month)
        if today.day in [1, 2]:
            results["monthly"] = await self.run_monthly_integration(
                agent_id=agent_id,
                tenant_id=tenant_id,
            )
        
        # Decay
        results["decay"] = await self.run_decay_cycle(
            tenant_id=tenant_id,
            agent_id=agent_id,
        )
        
        return results
