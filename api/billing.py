"""Stripe billing integration for OptiStock API.

Handles:
- Stripe product/price setup
- Checkout session creation
- Webhook event processing (idempotent)
- Subscription management
- API usage tracking (atomic)
- Billing portal
- Invoice history

Security:
- Never logs or stores Stripe API keys in plain text.
- Webhook signature verification is mandatory.
- All billing operations are idempotent.
- Usage tracking uses atomic upsert to prevent race conditions.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import stripe

from . import storage

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# Map env var names to Stripe price IDs (must be set at runtime)
_STRIPE_PRICES = {
    "starter": os.environ.get("STRIPE_PRICE_STARTER", ""),
    "professional": os.environ.get("STRIPE_PRICE_PROFESSIONAL", ""),
    "enterprise": os.environ.get("STRIPE_PRICE_ENTERPRISE", ""),
}

# Map billing tiers to internal tier names used by the optimizer
TIER_MAP = {
    "starter": "basic",
    "professional": "basic",
    "enterprise": "elite",
}

API_LIMITS = {
    "starter": 100,
    "professional": 1_000,
    "enterprise": -1,  # unlimited
}

# ── Stripe client setup ───────────────────────────────────────────────────
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
else:
    logger.warning("STRIPE_SECRET_KEY not set — billing will be non-functional")


# ── Product initialization ────────────────────────────────────────────────
async def ensure_stripe_products() -> None:
    """Create or lookup Stripe products/prices for each tier.

    Idempotent — safe to call on every startup.
    """
    if not STRIPE_SECRET_KEY:
        logger.info("Stripe not configured, skipping product setup")
        return

    products_config = [
        {
            "name": "OptiStock Starter",
            "description": "100 API calls/month, basic tier optimization",
            "unit_amount": 4900,  # $49.00 in cents
            "env_key": "starter",
        },
        {
            "name": "OptiStock Professional",
            "description": "1,000 API calls/month, basic tier optimization",
            "unit_amount": 14900,  # $149.00
            "env_key": "professional",
        },
        {
            "name": "OptiStock Enterprise",
            "description": "Unlimited API calls, elite tier optimization",
            "unit_amount": 39900,  # $399.00
            "env_key": "enterprise",
        },
    ]

    for config in products_config:
        price_id = _STRIPE_PRICES.get(config["env_key"])
        if price_id:
            # Price ID already configured — verify it exists
            try:
                stripe.Price.retrieve(price_id)
                logger.info(f"Verified existing price for {config['env_key']}: {price_id}")
            except stripe.error.InvalidRequestError:
                logger.error(
                    f"Configured price ID {price_id} for {config['env_key']} not found"
                )
            continue

        # Try to find existing price by product name
        try:
            products = stripe.Product.list(
                active=True,
                limit=100,
            )
            for product in products.auto_paging_iter():
                if product["name"] == config["name"]:
                    prices = stripe.Price.list(product=product["id"], active=True, limit=1)
                    price_list = list(prices.auto_paging_iter())
                    if price_list:
                        price_id = price_list[0]["id"]
                        _STRIPE_PRICES[config["env_key"]] = price_id
                        logger.info(
                            f"Found existing price for {config['env_key']}: {price_id}"
                        )
                        break
            if _STRIPE_PRICES.get(config["env_key"]):
                continue
        except Exception as e:
            logger.warning(f"Error listing Stripe products: {e}")

        # Create new product + price
        try:
            product = stripe.Product.create(
                name=config["name"],
                description=config["description"],
            )
            price = stripe.Price.create(
                product=product.id,
                unit_amount=config["unit_amount"],
                currency="usd",
                recurring={"interval": "month"},
            )
            _STRIPE_PRICES[config["env_key"]] = price.id
            logger.info(
                f"Created {config['env_key']} price: {price.id} "
                f"(${config['unit_amount'] / 100:.2f}/mo)"
            )
        except stripe.error.StripeError as e:
            logger.error(f"Failed to create Stripe product {config['env_key']}: {e}")


# ── Checkout ──────────────────────────────────────────────────────────────
async def create_checkout_session(
    user_id: str,
    email: str,
    price_id: str,
    success_url: str,
    cancel_url: str,
) -> dict:
    """Create a Stripe Checkout session for subscription signup.

    Returns {checkout_url, session_id}.
    Raises stripe.error.StripeError on failure.
    """
    session = stripe.checkout.Session.create(
        mode="subscription",
        customer_email=email,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={"user_id": user_id},
    )
    logger.info(f"Created checkout session {session.id} for user {user_id}")
    return {"checkout_url": session.url, "session_id": session.id}


# ── Webhook handling ──────────────────────────────────────────────────────
def verify_webhook_signature(payload: bytes, sig_header: str) -> dict:
    """Verify Stripe webhook signature.

    Raises stripe.error.SignatureVerificationError if invalid.
    Returns the parsed event dict.
    """
    if not STRIPE_WEBHOOK_SECRET:
        raise ValueError("STRIPE_WEBHOOK_SECRET not configured")

    event = stripe.Webhook.construct_event(
        payload, sig_header, STRIPE_WEBHOOK_SECRET
    )
    return event


async def handle_webhook_event(event: dict) -> None:
    """Process a verified Stripe webhook event.

    Idempotent — handles duplicate events safely.
    """
    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})
    event_id = event.get("id", "unknown")

    try:
        if event_type == "checkout.session.completed":
            await _handle_checkout_completed(data)
        elif event_type == "customer.subscription.updated":
            await _handle_subscription_updated(data)
        elif event_type == "customer.subscription.deleted":
            await _handle_subscription_deleted(data)
        elif event_type == "invoice.payment_failed":
            await _handle_invoice_payment_failed(data)
        elif event_type == "invoice.payment_succeeded":
            await _handle_invoice_payment_succeeded(data)
        else:
            logger.info(f"Ignored webhook event type: {event_type}")
    except Exception as e:
        # Log but don't raise — Stripe will retry
        logger.error(f"Error handling webhook event {event_type} ({event_id}): {e}")


async def _handle_checkout_completed(data: dict) -> None:
    """Activate subscription after successful checkout."""
    user_id = data.get("metadata", {}).get("user_id")
    stripe_sub_id = data.get("subscription")
    stripe_customer_id = data.get("customer")

    if not user_id or not stripe_sub_id:
        logger.warning("checkout.session.completed missing user_id or subscription")
        return

    # Fetch subscription details
    try:
        subscription = stripe.Subscription.retrieve(stripe_sub_id)
    except stripe.error.StripeError as e:
        logger.error(f"Failed to retrieve subscription {stripe_sub_id}: {e}")
        return

    # Determine tier from price
    price_id = None
    if subscription.get("items", {}).get("data"):
        price_id = subscription["items"]["data"][0].get("price", {}).get("id")

    tier = "unknown"
    for t, pid in _STRIPE_PRICES.items():
        if pid == price_id:
            tier = t
            break

    current_period_start = None
    current_period_end = None
    if subscription.get("current_period_start"):
        current_period_start = datetime.fromtimestamp(
            subscription["current_period_start"], tz=timezone.utc
        ).isoformat()
    if subscription.get("current_period_end"):
        current_period_end = datetime.fromtimestamp(
            subscription["current_period_end"], tz=timezone.utc
        ).isoformat()

    await storage.create_subscription(
        user_id=user_id,
        stripe_sub_id=stripe_sub_id,
        stripe_customer_id=stripe_customer_id,
        price_id=price_id,
        status="active",
        tier=tier,
        current_period_start=current_period_start,
        current_period_end=current_period_end,
    )

    # Reset usage for new subscription period
    if stripe_customer_id:
        await storage.reset_monthly_usage(stripe_customer_id)

    logger.info(f"Activated subscription for user {user_id}, tier {tier}")


async def _handle_subscription_updated(data: dict) -> None:
    """Handle subscription plan changes."""
    stripe_sub_id = data.get("id")
    if not stripe_sub_id:
        return

    status = data.get("status", "active")
    price_id = None
    if data.get("items", {}).get("data"):
        price_id = data["items"]["data"][0].get("price", {}).get("id")

    current_period_start = None
    current_period_end = None
    if data.get("current_period_start"):
        current_period_start = datetime.fromtimestamp(
            data["current_period_start"], tz=timezone.utc
        ).isoformat()
    if data.get("current_period_end"):
        current_period_end = datetime.fromtimestamp(
            data["current_period_end"], tz=timezone.utc
        ).isoformat()

    # Determine tier
    tier = "unknown"
    for t, pid in _STRIPE_PRICES.items():
        if pid == price_id:
            tier = t
            break

    await storage.update_subscription_status(
        stripe_sub_id=stripe_sub_id,
        status=status,
        price_id=price_id,
        tier=tier,
        current_period_start=current_period_start,
        current_period_end=current_period_end,
        cancel_at_period_end=data.get("cancel_at_period_end", False),
    )

    # Reset usage if period changed
    customer_id = data.get("customer")
    if customer_id and current_period_start:
        await storage.reset_monthly_usage(customer_id)

    logger.info(f"Updated subscription {stripe_sub_id}: status={status}, tier={tier}")


async def _handle_subscription_deleted(data: dict) -> None:
    """Cancel subscription."""
    stripe_sub_id = data.get("id")
    if not stripe_sub_id:
        return

    await storage.update_subscription_status(
        stripe_sub_id=stripe_sub_id,
        status="canceled",
    )
    logger.info(f"Canceled subscription {stripe_sub_id}")


async def _handle_invoice_payment_failed(data: dict) -> None:
    """Record failed payment."""
    await _record_invoice(data, status="failed")


async def _handle_invoice_payment_succeeded(data: dict) -> None:
    """Record successful payment."""
    await _record_invoice(data, status="paid")


async def _record_invoice(data: dict, status: str) -> None:
    """Store invoice record (idempotent by stripe_invoice_id)."""
    stripe_invoice_id = data.get("id")
    if not stripe_invoice_id:
        return

    amount = data.get("amount_due", 0) or 0
    amount_dollars = amount / 100.0

    customer_id = data.get("customer")
    invoice_url = data.get("hosted_invoice_url", "")
    created = data.get("created", 0)
    date_str = ""
    if created:
        date_str = datetime.fromtimestamp(created, tz=timezone.utc).isoformat()

    # Find user_id from subscription
    user_id = ""
    subscription_id = data.get("subscription")
    if subscription_id:
        sub = await storage.get_subscription_by_stripe_sub_id(subscription_id)
        if sub:
            user_id = sub["user_id"]

    await storage.create_invoice(
        user_id=user_id,
        stripe_invoice_id=stripe_invoice_id,
        amount=amount_dollars,
        status=status,
        date=date_str,
        url=invoice_url,
    )
    logger.info(f"Recorded invoice {stripe_invoice_id}: {status} (${amount_dollars:.2f})")


# ── Subscription management ──────────────────────────────────────────────
async def get_subscription(user_id: str) -> Optional[dict]:
    """Get current subscription for a user."""
    return await storage.get_subscription(user_id)


async def cancel_subscription(user_id: str) -> dict:
    """Cancel subscription at end of billing period.

    Returns {status, cancel_at, message}.
    """
    sub = await storage.get_subscription(user_id)
    if not sub:
        return {"status": "error", "message": "No active subscription found"}

    if sub["status"] == "canceled":
        return {"status": "error", "message": "Subscription already canceled"}

    if sub["cancel_at_period_end"]:
        return {
            "status": "already_canceled",
            "cancel_at": sub["current_period_end"],
            "message": "Subscription already set to cancel at period end",
        }

    try:
        stripe.Subscription.modify(
            sub["stripe_sub_id"],
            cancel_at_period_end=True,
        )
    except stripe.error.StripeError as e:
        logger.error(f"Failed to cancel subscription {sub['stripe_sub_id']}: {e}")
        return {"status": "error", "message": "Failed to cancel subscription"}

    await storage.update_subscription_status(
        stripe_sub_id=sub["stripe_sub_id"],
        cancel_at_period_end=True,
    )

    return {
        "status": "success",
        "cancel_at": sub["current_period_end"],
        "message": "Subscription will be canceled at end of billing period",
    }


async def update_subscription(user_id: str, new_tier: str) -> dict:
    """Upgrade or downgrade subscription plan.

    Returns {status, message}.
    """
    price_id = _STRIPE_PRICES.get(new_tier)
    if not price_id:
        return {"status": "error", "message": f"Invalid tier: {new_tier}"}

    sub = await storage.get_subscription(user_id)
    if not sub:
        return {"status": "error", "message": "No active subscription found"}

    if sub["price_id"] == price_id:
        return {"status": "error", "message": "Already on this plan"}

    try:
        stripe.Subscription.modify(
            sub["stripe_sub_id"],
            items=[
                {
                    "id": sub["stripe_sub_id"],
                    "price": price_id,
                }
            ],
            proration_behavior="always_invoice",
        )
    except stripe.error.StripeError as e:
        logger.error(
            f"Failed to update subscription {sub['stripe_sub_id']} to {new_tier}: {e}"
        )
        return {"status": "error", "message": f"Failed to update subscription: {str(e)}"}

    tier = TIER_MAP.get(new_tier, new_tier)
    await storage.update_subscription_status(
        stripe_sub_id=sub["stripe_sub_id"],
        price_id=price_id,
        tier=tier,
    )

    return {
        "status": "success",
        "message": f"Subscription updated to {new_tier}",
    }


# ── Billing portal ────────────────────────────────────────────────────────
async def create_billing_portal_session(customer_id: str) -> dict:
    """Create a Stripe Customer Portal session.

    Returns {portal_url}.
    """
    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=os.environ.get("BILLING_PORTAL_RETURN_URL", ""),
    )
    return {"portal_url": session.url}


# ── Invoice history ───────────────────────────────────────────────────────
async def get_invoices(user_id: str) -> list[dict]:
    """Get invoice history for a user."""
    return await storage.get_invoices(user_id)


# ── Usage tracking ────────────────────────────────────────────────────────
async def record_api_call(api_key: str) -> None:
    """Record an API call for usage tracking (fire-and-forget).

    Uses atomic upsert to prevent race conditions.
    Never raises — failures are logged silently.
    """
    try:
        await storage.record_api_call(api_key)
    except Exception as e:
        # Don't let billing tracking break API responses
        logger.error(f"Failed to record API call for usage tracking: {e}")


async def get_usage(user_id: str) -> dict:
    """Get current API usage for a user.

    Returns {current_usage, limit, period_start, period_end, tier}.
    """
    usage = await storage.get_usage(user_id)
    if not usage:
        return {
            "current_usage": 0,
            "limit": 100,  # default starter limit
            "period_start": "",
            "period_end": "",
            "tier": "starter",
        }

    limit = API_LIMITS.get(usage.get("tier", "starter"), 100)
    return {
        "current_usage": usage["usage_count"],
        "limit": limit,
        "period_start": usage["period_start"],
        "period_end": usage["period_end"],
        "tier": usage.get("tier", "starter"),
    }


async def check_usage_limit(user_id: str) -> bool:
    """Check if user is under their API usage limit.

    Returns True if under limit (or unlimited), False if exceeded.
    """
    usage = await storage.get_usage(user_id)
    if not usage:
        return True  # no subscription, allow (will be handled by auth)

    tier = usage.get("tier", "starter")
    limit = API_LIMITS.get(tier, 100)

    if limit < 0:  # unlimited
        return True

    return usage["usage_count"] < limit
