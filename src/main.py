import os
from contextlib import asynccontextmanager
from logging import getLogger

from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from .agent import agent
from .flight import agent as flight_agent
from .hotel import agent as hotel_agent
from .server.error import init_error_handlers
from .server.middleware import init_middleware

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Server running on port {os.getenv('PORT', 80)}")
    try:
        # Initialize the SDK
        sdk = await get_sdk()

        # Store in app state
        app.state.sdk = sdk

        # Add CopilotKit endpoint
        add_fastapi_endpoint(app, sdk, "/copilotkit")

        yield
        logger.info("Server shutting down")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise


# Create the SDK after the graph is initialized
async def get_sdk():
    supervisor = await agent()
    flight = await flight_agent()
    hotel = await hotel_agent()
    sdk = CopilotKitRemoteEndpoint(
        agents=[
            LangGraphAgent(
                name="supervisor", description="Book a trip", graph=supervisor
            ),
            LangGraphAgent(name="hotel-agent", description="Book a hotel", graph=hotel),
            LangGraphAgent(
                name="flight-agent", description="Book a flight", graph=flight
            ),
        ],
    )
    return sdk


app = FastAPI(lifespan=lifespan)


@app.post("/")
def root():
    # Headers to disable proxy/CDN buffering (CloudFront, nginx, etc.)
    return StreamingResponse(
        "This agent is meant to be used with CopilotKit.\n"
        "You can follow this documentation to use it: "
        "https://docs.blaxel.ai/Agents/Integrate-in-apps/CopilotKit#copilotkit-integration",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


init_error_handlers(app)
init_middleware(app)

FastAPIInstrumentor.instrument_app(app, exclude_spans=["receive", "send"])
