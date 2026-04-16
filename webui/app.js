(function() {
    'use strict';

    // === Config ===
    const API_BASE = window.OPTISTOCK_API_URL || '';

    // === Smooth Scroll ===
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                // Close mobile nav if open
                document.querySelector('.nav-links')?.classList.remove('open');
            }
        });
    });

    // === Animated Counters ===
    function animateCounters() {
        document.querySelectorAll('.stat-num').forEach(el => {
            const target = parseInt(el.dataset.target);
            if (isNaN(target)) return;
            const suffix = el.textContent.replace(/[0-9]/g, '');
            let current = 0;
            const step = Math.max(1, Math.floor(target / 40));
            const timer = setInterval(() => {
                current += step;
                if (current >= target) {
                    current = target;
                    clearInterval(timer);
                }
                el.textContent = current + suffix;
            }, 30);
        });
    }

    // Run counters when hero is visible
    const heroObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounters();
                heroObserver.disconnect();
            }
        });
    }, { threshold: 0.3 });
    const heroEl = document.querySelector('.hero-stats');
    if (heroEl) heroObserver.observe(heroEl);

    // === Copy to Clipboard ===
    window.copyCode = function(btn) {
        const code = btn.closest('.api-code-block').querySelector('code').textContent;
        navigator.clipboard.writeText(code).then(() => {
            btn.textContent = 'Copied!';
            setTimeout(() => btn.textContent = 'Copy', 2000);
        });
    };

    // === Demo Form ===
    function generateHistoricalData(days) {
        const data = [];
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - days);
        for (let i = 0; i < days; i++) {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i);
            const dow = date.getDay();
            const base = 6;
            const seasonal = Math.sin((i / 365) * 2 * Math.PI) * 2;
            const noise = Math.random() * 4 - 2;
            const weekendBonus = (dow === 0 || dow === 6) ? 2 : 0;
            const qty = Math.max(0, Math.round(base + seasonal + noise + weekendBonus));
            data.push({
                date: date.toISOString().split('T')[0],
                quantity: qty,
                available: 200
            });
        }
        return data;
    }

    window.runDemo = async function() {
        const btn = document.getElementById('demo-btn');
        const resultsEl = document.getElementById('demo-results');

        // Read form values
        const itemId = document.getElementById('demo-item-id').value || 'SKU-001';
        const cost = parseFloat(document.getElementById('demo-cost').value) || 12.50;
        const price = parseFloat(document.getElementById('demo-price').value) || 29.99;
        const stock = parseInt(document.getElementById('demo-stock').value) || 150;
        const onOrder = parseInt(document.getElementById('demo-on-order').value) || 0;
        const leadTime = parseInt(document.getElementById('demo-lead-time').value) || 14;
        const orderFreq = parseInt(document.getElementById('demo-order-freq').value) || 7;
        const tier = document.getElementById('demo-tier').value || 'basic';
        const historicalRaw = document.getElementById('demo-historical').value;
        const apiKey = document.getElementById('demo-api-key')?.value?.trim() || '';

        // Parse historical data
        let historicalData;
        try {
            // Try parsing as JSON array of objects first
            const parsed = JSON.parse(historicalRaw);
            if (Array.isArray(parsed) && parsed.length > 0 && typeof parsed[0] === 'object') {
                historicalData = parsed;
            } else {
                // Try as array of numbers
                const numbers = Array.isArray(parsed) ? parsed : historicalRaw.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
                historicalData = generateHistoricalDataFromNumbers(numbers, itemId);
            }
        } catch (e) {
            // Parse as comma-separated numbers
            const numbers = historicalRaw.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
            if (numbers.length === 0) {
                // Generate default data
                historicalData = generateHistoricalData(90);
            } else {
                historicalData = generateHistoricalDataFromNumbers(numbers, itemId);
            }
        }

        const requestBody = {
            items: [{
                item_id: itemId,
                cost: cost,
                sale_price: price,
                current_available: stock,
                on_order_qty: onOrder,
                back_order_qty: 0,
                lead_time_days: leadTime,
                order_frequency_days: orderFreq,
                payment_terms_days: 30,
                sales_terms_days: 30,
                length: 1.0,
                width: 1.0,
                height: 1.0,
                historical_data: historicalData
            }],
            tier: tier,
            cost_of_capital: 0.14
        };

        // Show loading
        btn.disabled = true;
        btn.textContent = 'Optimizing...';
        resultsEl.innerHTML = '<div class="results-placeholder"><p>⏳ Running optimization...</p></div>';

        try {
            const response = await fetch(`${API_BASE}/v1/optimize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(apiKey && { 'X-API-Key': apiKey })
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: 'Unknown error' }));
                throw new Error(error.error || `HTTP ${response.status}`);
            }

            const data = await response.json();
            displayResults(data, resultsEl);

        } catch (error) {
            resultsEl.innerHTML = `
                <div class="result-card" style="border-left-color: #ef4444;">
                    <h4>Error</h4>
                    <p style="color: #ef4444;">${error.message}</p>
                    <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 8px;">
                        Make sure the API server is running. For local testing, run:<br>
                        <code>OPTISTOCK_API_KEYS="" uvicorn modal_app:fastapi_app --reload --port 8000</code>
                    </p>
                </div>
            `;
        } finally {
            btn.disabled = false;
            btn.textContent = '⚡ Optimize Now';
        }
    };

    function generateHistoricalDataFromNumbers(numbers, itemId) {
        const data = [];
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - numbers.length);
        for (let i = 0; i < numbers.length; i++) {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i);
            data.push({
                date: date.toISOString().split('T')[0],
                quantity: numbers[i],
                available: 200
            });
        }
        return data;
    }

    function displayResults(data, container) {
        if (!data.items || data.items.length === 0) {
            container.innerHTML = '<div class="results-placeholder"><p>No results returned</p></div>';
            return;
        }

        const item = data.items[0];
        container.innerHTML = `
            <h3 style="margin-bottom: 16px; font-size: 1.1rem;">Optimization Results</h3>
            <div class="result-grid">
                <div class="result-card">
                    <h4>Recommended Order</h4>
                    <div class="value">${item.recommended_order_qty.toLocaleString()} units</div>
                </div>
                <div class="result-card">
                    <h4>Expected Profit</h4>
                    <div class="value">$${item.expected_profit.toFixed(2)}</div>
                </div>
                <div class="result-card">
                    <h4>Optimal Stock Level</h4>
                    <div class="value">${item.optimal_outp.toLocaleString()} units</div>
                </div>
                <div class="result-card">
                    <h4>Avg Daily Sales</h4>
                    <div class="value">${item.expected_daily_sales.toFixed(1)}</div>
                </div>
                <div class="result-card">
                    <h4>Avg Inventory</h4>
                    <div class="value">${item.expected_avg_inventory.toFixed(1)} units</div>
                </div>
                <div class="result-card">
                    <h4>Profit per Cube</h4>
                    <div class="value">$${item.profit_per_cube.toFixed(2)}</div>
                </div>
            </div>
            <div class="result-meta">
                <span>Demand Source: <code>${item.demand_source}</code></span>
                <span>Compute Time: <code>${data.compute_time_ms.toFixed(0)}ms</code></span>
                ${item.warnings && item.warnings.length > 0 ? `<span style="color: #f59e0b;">Warnings: ${item.warnings.join(', ')}</span>` : ''}
            </div>
        `;
    }

    // === Load sample data button (if exists) ===
    document.addEventListener('DOMContentLoaded', function() {
        // Trigger counter animation if hero is already visible
        const heroSection = document.querySelector('.hero');
        if (heroSection) {
            const rect = heroSection.getBoundingClientRect();
            if (rect.top < window.innerHeight) {
                animateCounters();
            }
        }
    });
})();
