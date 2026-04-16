"""Billing routes for OptiStock API — Stripe integration endpoints."""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi import Header
from pydantic import BaseModel

from . import billing, user_auth
from .user_auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/billing", tags=["billing"])


class CheckoutRequest(BaseModel):
    price_tier: str


class UpdateSubscriptionRequest(BaseModel):
    new_tier: str


@router.post("/checkout")
async def create_checkout(
    body: CheckoutRequest,
    user: dict = Depends(get_current_user),
):
    """Create a Stripe checkout session for the given tier."""
    result = await billing.create_checkout_session(
        user_id=user["id"],
        email=user["email"],
        price_id=body.price_tier,
        success_url=os.environ.get("CHECKOUT_SUCCESS_URL", "/static/dashboard.html?tab=billing&status=success"),
        cancel_url=os.environ.get("CHECKOUT_CANCEL_URL", "/static/dashboard.html?tab=billing&status=cancelled"),
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/webhook")
async def stripe_webhook(request: Request, stripe_signature: str = Header(None)):
    """Handle Stripe webhook events."""
    body = await request.body()
    try:
        event = billing.verify_webhook_signature(body, stripe_signature)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})

    await billing.handle_webhook_event(event)
    return {"status": "success"}


@router.get("/subscription")
async def get_subscription(user: dict = Depends(get_current_user)):
    """Get current subscription details."""
    sub = await billing.get_subscription(user["id"])
    if not sub:
        return {"has_subscription": False}
    return {"has_subscription": True, **sub}


@router.post("/cancel")
async def cancel_subscription(user: dict = Depends(get_current_user)):
    """Cancel subscription at end of billing period."""
    result = await billing.cancel_subscription(user["id"])
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/update")
async def update_subscription(
    body: UpdateSubscriptionRequest,
    user: dict = Depends(get_current_user),
):
    """Upgrade or downgrade subscription plan."""
    result = await billing.update_subscription(user["id"], body.new_tier)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/portal")
async def create_portal(user: dict = Depends(get_current_user)):
    """Create Stripe Customer Portal session for billing management."""
    sub = await billing.get_subscription(user["id"])
    if not sub or not sub.get("stripe_customer_id"):
        raise HTTPException(status_code=400, detail={"error": "No billing account found"})
    return await billing.create_billing_portal_session(sub["stripe_customer_id"])


@router.get("/invoices")
async def get_invoices(user: dict = Depends(get_current_user)):
    """Get invoice history."""
    return await billing.get_invoices(user["id"])


@router.get("/usage")
async def get_usage(user: dict = Depends(get_current_user)):
    """Get current API usage stats."""
    return await billing.get_usage(user["id"])
