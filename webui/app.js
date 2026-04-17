(function() {
    'use strict';

    // === Config ===
    const API_BASE = window.OPTISTOCK_API_URL || '';

    // === Mobile Nav Toggle ===
    const navToggle = document.getElementById('nav-toggle');
    if (navToggle) {
        navToggle.addEventListener('click', () => {
            const navLinks = document.getElementById('nav-links');
            if (navLinks) navLinks.classList.toggle('open');
            navToggle.setAttribute('aria-expanded', navLinks?.classList.contains('open') ? 'true' : 'false');
        });
    }

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
        const backOrder = parseInt(document.getElementById('demo-back-order').value) || 0;
        const leadTime = parseInt(document.getElementById('demo-lead-time').value) || 14;
        const orderFreq = parseInt(document.getElementById('demo-order-freq').value) || 7;
        const coc = parseFloat(document.getElementById('demo-coc').value) || 5;
        const length = parseFloat(document.getElementById('demo-length').value) || 1.0;
        const width = parseFloat(document.getElementById('demo-width').value) || 1.0;
        const height = parseFloat(document.getElementById('demo-height').value) || 1.0;
        const paymentTerms = parseInt(document.getElementById('demo-payment-terms').value) || 30;
        const salesTerms = parseInt(document.getElementById('demo-sales-terms').value) || 30;
        const netProfitRaw = document.getElementById('demo-net-profit')?.value?.trim();
        const netProfit = netProfitRaw ? parseFloat(netProfitRaw) : null;
        const tier = document.getElementById('demo-tier').value || 'basic';
        const historicalRaw = document.getElementById('demo-historical').value;

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
                back_order_qty: backOrder,
                lead_time_days: leadTime,
                order_frequency_days: orderFreq,
                payment_terms_days: paymentTerms,
                sales_terms_days: salesTerms,
                length: length,
                width: width,
                height: height,
                historical_data: historicalData,
                ...(netProfit !== null && { net_profit_per_unit: netProfit })
            }],
            tier: tier,
            cost_of_capital: coc / 100  // convert percentage to decimal
        };

        // Show loading
        btn.disabled = true;
        btn.textContent = 'Optimizing...';
        resultsEl.innerHTML = '<div class="results-placeholder"><p>⏳ Running optimization...</p></div>';

        try {
            const response = await fetch(`${API_BASE}/v1/optimize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
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
        const outpCurve = item.outp_curve || [];
        const curveChart = outpCurve.length > 0 ? `
            <div class="result-chart">
                <h4 style="margin-bottom: 12px;">Profit vs Order Up to Point</h4>
                <canvas id="outp-chart" height="200"></canvas>
            </div>
        ` : '';

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
                    <h4>Optimal Order Up to Point</h4>
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
            ${curveChart}
            <div class="result-meta">
                <span>Demand Source: <code>${item.demand_source}</code></span>
                <span>Compute Time: <code>${data.compute_time_ms.toFixed(0)}ms</code></span>
                ${item.warnings && item.warnings.length > 0 ? `<span style="color: #f59e0b;">Warnings: ${item.warnings.join(', ')}</span>` : ''}
            </div>
        `;

        // Draw OUTP profit curve chart
        if (outpCurve.length > 0) {
            setTimeout(() => drawOutpChart(outpCurve, item.optimal_outp), 50);
        }
    }

    function drawOutpChart(curve, optimalOutp) {
        const canvas = document.getElementById('outp-chart');
        if (!canvas) return;
        const dpr = window.devicePixelRatio || 1;
        const displayWidth = canvas.offsetWidth;
        const displayHeight = 240;
        canvas.width = displayWidth * dpr;
        canvas.height = displayHeight * dpr;
        canvas.style.height = displayHeight + 'px';
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        const width = displayWidth;
        const height = displayHeight;

        const pad = { top: 28, right: 24, bottom: 42, left: 60 };
        const cw = width - pad.left - pad.right;
        const ch = height - pad.top - pad.bottom;

        const xs = curve.map(p => p.outp);
        const ys = curve.map(p => p.profit);
        const xMin = xs[0], xMax = xs[xs.length - 1];
        const yMin = Math.min(0, ...ys);
        const yMax = Math.max(...ys);
        const rx = xMax - xMin || 1;
        const ry = yMax - yMin || 1;
        const xS = v => pad.left + ((v - xMin) / rx) * cw;
        const yS = v => pad.top + ch - ((v - yMin) / ry) * ch;

        // Background
        ctx.fillStyle = '#fafbfc';
        ctx.fillRect(0, 0, width, height);

        // Subtle grid
        ctx.strokeStyle = '#e8ecf0';
        ctx.lineWidth = 0.5;
        const yTicks = niceScale(yMin, yMax, 5);
        yTicks.forEach(v => {
            const y = yS(v);
            ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
        });

        // Zero line
        if (yMin < 0) {
            ctx.strokeStyle = '#c0c5cc';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 3]);
            ctx.beginPath(); ctx.moveTo(pad.left, yS(0)); ctx.lineTo(width - pad.right, yS(0)); ctx.stroke();
            ctx.setLineDash([]);
        }

        // Gradient fill under curve
        const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
        grad.addColorStop(0, 'rgba(99,102,241,0.18)');
        grad.addColorStop(1, 'rgba(99,102,241,0.01)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.moveTo(xS(xs[0]), yS(0 > yMin ? 0 : yMin));
        curve.forEach(p => ctx.lineTo(xS(p.outp), yS(p.profit)));
        ctx.lineTo(xS(xs[xs.length - 1]), yS(0 > yMin ? 0 : yMin));
        ctx.closePath();
        ctx.fill();

        // Smooth curve
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2.5;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        curve.forEach((p, i) => {
            const x = xS(p.outp), y = yS(p.profit);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Optimal point
        const optPt = curve.find(p => p.outp === optimalOutp);
        if (optPt) {
            const ox = xS(optPt.outp), oy = yS(optPt.profit);
            // Glow
            ctx.beginPath(); ctx.arc(ox, oy, 10, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(239,68,68,0.15)'; ctx.fill();
            // Dot
            ctx.beginPath(); ctx.arc(ox, oy, 5, 0, Math.PI * 2);
            ctx.fillStyle = '#ef4444'; ctx.fill();
            ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(ox, oy, 5, 0, Math.PI * 2); ctx.stroke();
            // Label
            ctx.fillStyle = '#1e293b';
            ctx.font = 'bold 11px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`Optimal: ${optPt.outp} units → $${optPt.profit.toFixed(0)}`, ox, oy - 16);
        }

        // X axis labels
        ctx.fillStyle = '#6b7280';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        const xTicks = niceScale(xMin, xMax, 6);
        xTicks.forEach(v => {
            ctx.fillText(Math.round(v), xS(v), height - pad.bottom + 20);
        });
        ctx.fillText('Order Up to Point (units)', width / 2, height - 6);

        // Y axis labels
        ctx.textAlign = 'right';
        yTicks.forEach(v => {
            ctx.fillText('$' + Math.round(v), pad.left - 8, yS(v) + 4);
        });
    }

    function niceScale(min, max, ticks) {
        const range = max - min || 1;
        const rough = range / ticks;
        const mag = Math.pow(10, Math.floor(Math.log10(rough)));
        const residual = rough / mag;
        let nice;
        if (residual <= 1.5) nice = mag;
        else if (residual <= 3) nice = 2 * mag;
        else if (residual <= 7) nice = 5 * mag;
        else nice = 10 * mag;
        const result = [];
        let v = Math.ceil(min / nice) * nice;
        while (v <= max) { result.push(v); v += nice; }
        return result;
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
