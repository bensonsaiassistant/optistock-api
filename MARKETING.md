# OptiStock — Go-to-Market Plan

## Executive Summary

OptiStock is a profit-maximizing inventory purchase optimization API built for Amazon FBA sellers who need to answer one question: **"How many units should I order right now?"**

We use Monte Carlo simulation (100K-day inventory modeling), ML demand forecasting, and lead time prediction to recommend exact order quantities that maximize profit while accounting for demand uncertainty, supply chain variability, storage costs, and capital costs.

**Target:** Small-to-medium Amazon FBA sellers ($500K–$10M annual revenue, 50–5,000 SKUs).

---

## Market Problem

Amazon sellers are drowning in inventory decisions:

1. **Overstocking** → FBA storage fees eat margins, long-term storage penalties kick in after 365 days
2. **Understocking** → Lost sales, ranking drops, competitors take your buy box
3. **Guesswork ordering** → Most sellers use gut feel or simple spreadsheets
4. **No lead time modeling** → Suppliers are unreliable, but nobody accounts for it
5. **Capital is expensive** → 14%+ APR on credit lines means every dollar in inventory has a real cost

**Existing solutions:** SellerApp, Helium10, JungleScout — all focus on product research and keyword optimization. None optimize *how many units to order* using simulation.

---

## Product Positioning

**"The inventory optimizer for Amazon sellers who want to stop guessing."**

| Competitor | What they do | What they DON'T do |
|-----------|-------------|-------------------|
| Helium10 | Product research, keyword tools | Inventory optimization |
| SellerApp | Analytics, PPC management | Profit-maximizing order quantities |
| RestockPro | Basic reorder alerts | Monte Carlo simulation, ML forecasting |
| Spreadsheets | Manual calculations | Anything automated |

**OptiStock's moat:** Physics-grade simulation + ML forecasting in an API. Nobody else is doing this.

---

## Three-Tier Pricing (SaaS)

### Starter — $49/month
- Up to 50 SKUs
- Basic tier (rolling average demand)
- Single-item optimization
- Email support
- **Target:** New sellers, < $500K/year revenue

### Professional — $149/month
- Up to 500 SKUs
- Premium tier (ML demand forecasting with XGBoost)
- Batch optimization (3–10 items at once)
- API access + dashboard
- Priority support
- **Target:** Growing sellers, $500K–$5M/year revenue

### Enterprise — $399/month
- Unlimited SKUs
- Elite tier (ML demand + ML lead time prediction)
- Full batch parallel optimization
- Custom integration support
- Dedicated account manager
- **Target:** Established sellers, $5M+/year revenue, brand aggregators

---

## Distribution Channels

### 1. Amazon Seller Communities (Zero CAC)
- **Reddit:** r/FulfillmentByAmazon, r/AmazonSeller — answer questions, share case studies
- **Facebook Groups:** "Amazon FBA High Rollers", "FBA Warriors" — 50K+ members each
- **YouTube:** Partner with FBA educators (Dan Vas, Amazing Selling Machine alumni) for sponsored content
- **Discord/Slack:** Seller communities, e-commerce ops channels

**Content strategy:**
- "How I stopped guessing my Amazon inventory orders" — personal story format
- "The math behind optimal inventory levels" — educational, data-driven
- "Why your spreadsheet is costing you $5K/month in inventory waste" — provocative, specific

### 2. Direct Outreach (Low CAC)
- **Seller databases:** JungleScout/Helium10 users (public reviewer lists)
- **Amazon brand registry:** Target sellers with 50+ SKUs (they feel the pain most)
- **Trade shows:** Prosper Show, Seller Con, eTail West — booth or speaking slot
- **3PL partnerships:** Sellers using FBA prep centers and 3PLs need this

### 3. API-First Play (Viral Loop)
- **Free tier:** 100 API calls/month free — get developers hooked
- **Documentation-first:** Excellent docs at docs.optistock.ai — developers share internally
- **GitHub:** Open-source the simulation engine (not the API), build community credibility
- **Integrations:** Shopify, Amazon SP-API, ShipStation — plug into existing workflows

### 4. Content Marketing (Medium CAC)
- **Blog:** "Inventory Intelligence" — data-driven posts about Amazon seller economics
- **Case studies:** Real seller results ("How Seller X reduced storage costs 34%")
- **Calculator tool:** Free "Should I Reorder?" calculator on landing page → lead capture
- **Webinars:** Monthly "Optimize Your Q4 Inventory" sessions

---

## Key Messaging

### For the Seller Who's Tired of Stockouts
> "You're losing $2,800/month in missed sales because your reorder timing is off. OptiStock tells you exactly when and how much to order — using the same Monte Carlo simulation that Fortune 500 companies pay $50K/year for."

### For the Analytical Seller
> "Your spreadsheet guesses. Our simulation models 100,000 days of inventory scenarios, accounts for demand variability and supplier lead times, and finds the exact order quantity that maximizes your profit. It's not a guess — it's math."

### For the Growing Brand
> "At 500 SKUs, you can't manage inventory by gut feel anymore. OptiStock optimizes your entire catalog in parallel, so you order what you need, when you need it, without overthinking every SKU."

---

## Launch Plan

### Phase 1: Pre-Launch (Weeks 1–4)
- [ ] Build landing page at optistock.ai with waitlist
- [ ] Create 3 case study videos showing real seller results
- [ ] Write 5 blog posts for SEO ("Amazon inventory optimization", "FBA storage fees calculator", etc.)
- [ ] Set up free API tier with 100 calls/month
- [ ] Join 10+ Amazon seller communities
- [ ] Build "Should I Reorder?" free calculator tool

### Phase 2: Soft Launch (Weeks 5–8)
- [ ] Launch to waitlist (target: 200 signups)
- [ ] Offer 50% off first 3 months for founding members
- [ ] Partner with 3 Amazon FBA YouTubers for sponsored reviews
- [ ] Collect testimonials and case studies
- [ ] Iterate on product based on user feedback
- [ ] Target: $2K MRR from founding members

### Phase 3: Growth (Weeks 9–16)
- [ ] Launch paid ads (Amazon seller Facebook groups, Reddit)
- [ ] Attend Prosper Show or Seller Con
- [ ] Publish "State of Amazon Inventory" annual report (PR hook)
- [ ] Build integrations: Shopify app, Amazon SP-API connector
- [ ] Launch referral program (1 month free for each referral)
- [ ] Target: $10K MRR

### Phase 4: Scale (Months 5–12)
- [ ] Hire 1 sales rep for enterprise outreach
- [ ] Build white-label version for 3PLs and agencies
- [ ] Launch API marketplace listing (RapidAPI, AWS Marketplace)
- [ ] Expand to Walmart Marketplace sellers
- [ ] Target: $50K MRR

---

## Revenue Projections

| Month | Users | MRR | Notes |
|-------|-------|-----|-------|
| 1–2 | 50 | $2,500 | Founding members |
| 3–4 | 150 | $7,500 | Soft launch momentum |
| 5–6 | 300 | $15,000 | Paid ads + content |
| 7–9 | 500 | $25,000 | Conference + integrations |
| 10–12 | 800 | $40,000 | Enterprise deals |

**Year 1 target:** $200K+ ARR with 800+ paying sellers.

---

## Competitive Advantages

1. **Technical moat:** Monte Carlo simulation + ML forecasting in an API. Nobody else has this.
2. **Speed:** Batch optimize 10 items in ~1 second vs. hours in a spreadsheet
3. **Accuracy:** ML demand forecasting with deviation guardrails — falls back to baselines when uncertain
4. **Elite tier:** The only tool that models lead time uncertainty using ML
5. **Developer-first:** API means it plugs into any workflow (Shopify, custom dashboards, automated reordering)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Amazon sellers don't care about optimization | Prove ROI: "Saves $2,800/month in wasted inventory" |
| ML models need too much data | Basic tier works with just 7 days of history |
| Competitors copy the approach | Move fast, build integrations, lock in customers |
| API complexity scares sellers | Simple dashboard UI on top of API, or partner with agencies |
| Seasonal demand patterns confuse models | Elite tier's ML handles seasonality automatically |

---

## Next Steps

1. **Build the landing page** — optistock.ai with waitlist + free calculator
2. **Create demo content** — 3 case study videos, 5 blog posts
3. **Join communities** — Reddit, Facebook groups, Discord
4. **Launch free API tier** — 100 calls/month, get developers hooked
5. **Collect early testimonials** — Founding members at 50% off for 3 months
6. **Iterate fast** — Use feedback to improve the product before scaling
