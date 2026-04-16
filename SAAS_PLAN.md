# OptiStock SaaS Implementation Plan

## Phase 1: User Management & Auth (Core)
- User signup/login with email + password
- API key generation per user
- Auth middleware for API endpoints
- SQLite user database (upgradeable to Postgres later)

## Phase 2: Stripe Billing
- Subscription plans matching our tiers ($49/$149/$399)
- Stripe Checkout integration
- Webhook handling for payments
- Usage tracking & billing

## Phase 3: Customer Dashboard
- Web UI for users to manage their account
- View API keys, usage stats, billing info
- Regenerate keys, upgrade/downgrade plans

## Phase 4: Legal & Polish
- Terms of Service, Privacy Policy
- Email verification
- Password reset
- Custom domain setup

## File Structure
```
api/
├── users.py          # User model, auth logic
├── billing.py        # Stripe integration, webhooks
├── storage.py        # Extended with user tables
└── auth.py           # Extended with user auth
webui/
├── dashboard.html    # Customer dashboard
├── signup.html       # Signup page
├── login.html        # Login page
├── pricing.html      # Pricing/checkout page
├── tos.html          # Terms of Service
├── privacy.html      # Privacy Policy
└── index.html        # Updated landing page
```

## Tech Choices
- **Auth**: Email + password with bcrypt hashing
- **Database**: SQLite (existing storage.py) with new user tables
- **Billing**: Stripe Checkout + webhooks
- **Frontend**: Extend existing vanilla HTML/CSS/JS
- **API Keys**: User-specific keys with rate limiting
