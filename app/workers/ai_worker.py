import asyncio
import structlog
from bullmq import Worker
from app.config.redis import get_clean_redis_url

logger = structlog.get_logger("vyaparsathi.ai.worker")

_worker: Worker | None = None


async def process_ai_job(job, job_token: str):
    logger.info("processing_bullmq_job", job_id=job.id, job_name=job.name, data=job.data)
    
    try:
        if job.name == "generate_forecast":
            from app.services.forecast_service import generate_forecast
            store_id = job.data.get("storeId", "default")
            timeframe = job.data.get("timeframe", "30d")
            # Execute forecasting service
            result = await generate_forecast(store_id=store_id, timeframe=timeframe)
            return {"status": "success", "result": result}

        elif job.name == "generate_excel_report":
            # Document generation job
            report_title = job.data.get("title", "Report")
            return {"status": "success", "report": report_title, "url": "generated_file.xlsx"}

        elif job.name == "ping_test":
            return {"pong": True, "message": "Python BullMQ Worker active!"}

        else:
            logger.warning("unknown_job_type", job_name=job.name)
            return {"status": "ignored", "reason": f"Unknown job name '{job.name}'"}

    except Exception as e:
        logger.error("bullmq_job_failed", job_id=job.id, error=str(e))
        raise e


async def start_python_worker() -> Worker | None:
    global _worker
    redis_url = get_clean_redis_url()
    if not redis_url:
        logger.warning("bullmq_worker_skipped_no_redis_url")
        return None

    try:
        # Initialize Python BullMQ worker connected to 'ai-tasks' queue
        _worker = Worker("ai-tasks", process_ai_job, {"connection": redis_url})
        logger.info("python_bullmq_worker_started", queue="ai-tasks")
        return _worker
    except Exception as e:
        logger.error("python_bullmq_worker_start_failed", error=str(e))
        return None


async def stop_python_worker():
    global _worker
    if _worker:
        try:
            await _worker.close()
            logger.info("python_bullmq_worker_stopped")
        except Exception as e:
            logger.error("python_bullmq_worker_stop_failed", error=str(e))
        finally:
            _worker = None


if __name__ == "__main__":
    async def main():
        worker = await start_python_worker()
        if worker:
            print("[Python BullMQ Worker] Listening on queue 'ai-tasks'...")
            # Keep process running
            while True:
                await asyncio.sleep(1)

    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Worker stopped.")
