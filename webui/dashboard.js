(function() {
    'use strict';

    const API = ''; // relative URLs since served by same FastAPI

    // === Auth Guard ===
    function getToken() { return localStorage.getItem('token'); }
    function getUser() { return JSON.parse(localStorage.getItem('user') || '{}'); }

    function checkAuth() {
        const token = getToken();
        if (!token) {
            window.location.href = '/static/login.html';
            return false;
        }
        return true;
    }

    // === Tab Switching ===
    window.switchTab = function(tabId, el) {
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        const tab = document.getElementById('tab-' + tabId);
        if (tab) tab.classList.add('active');
        if (el) el.classList.add('active');

        const titles = { overview: 'Overview', keys: 'API Keys', usage: 'Usage', billing: 'Billing', settings: 'Settings' };
        document.getElementById('page-title').textContent = titles[tabId] || 'Dashboard';

        // Load tab data
        if (tabId === 'keys') loadApiKeys();
        if (tabId === 'usage') loadUsage();
        if (tabId === 'billing') loadBilling();
        if (tabId === 'overview') loadOverview();
        if (tabId === 'settings') loadSettings();
    };

    // === API Calls ===
    async function apiCall(endpoint, options = {}) {
        const token = getToken();
        const headers = { 'Content-Type': 'application/json', ...(token && { 'Authorization': `Bearer ${token}` }) };
        const res = await fetch(`${API}${endpoint}`, { ...options, headers: { ...headers, ...options.headers } });
        if (res.status === 401) {
            localStorage.clear();
            window.location.href = '/static/login.html';
            throw new Error('Unauthorized');
        }
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail?.error || data.detail || data.error || 'API Error');
        return data;
    }

    // === Load Overview ===
    async function loadOverview() {
        try {
            const user = getUser();
            document.getElementById('user-greeting').textContent = `Welcome, ${user.name || 'User'}`;

            // Load stats in parallel
            const [usage, keys] = await Promise.allSettled([
                apiCall('/v1/billing/usage'),
                apiCall('/v1/keys')
            ]);

            if (usage.status === 'fulfilled' && usage.value) {
                document.getElementById('overview-plan').textContent = (usage.value.tier || 'free').charAt(0).toUpperCase() + (usage.value.tier || 'free').slice(1);
                document.getElementById('overview-usage').textContent = usage.value.current_usage || 0;
            }
            if (keys.status === 'fulfilled' && keys.value) {
                document.getElementById('overview-keys').textContent = keys.value.length || 0;
            }
        } catch (e) {
            console.error('Failed to load overview:', e);
        }
    }

    // === Load API Keys ===
    async function loadApiKeys() {
        try {
            const keys = await apiCall('/v1/keys');
            const listEl = document.getElementById('keys-list');

            if (!keys || keys.length === 0) {
                listEl.innerHTML = '<p class="empty-state">No API keys yet. Create one to get started.</p>';
                return;
            }

            listEl.innerHTML = keys.map(k => `
                <div class="key-item">
                    <div class="key-info">
                        <span class="key-name">${escapeHtml(k.name || 'Unnamed Key')}</span>
                        <span class="key-hash">osk-...${(k.key_hash || '').slice(-4)}</span>
                    </div>
                    <div>
                        <span class="key-status ${k.is_active ? 'active' : 'revoked'}">${k.is_active ? 'Active' : 'Revoked'}</span>
                        <span class="key-meta">${new Date(k.created_at).toLocaleDateString()}</span>
                        ${k.is_active ? `<button class="btn btn-danger" style="margin-left:8px;padding:4px 12px;font-size:0.8rem;" onclick="revokeKey('${k.id}')">Revoke</button>` : ''}
                    </div>
                </div>
            `).join('');
        } catch (e) {
            console.error('Failed to load keys:', e);
        }
    }

    // === Create Key Modal ===
    window.showCreateKeyModal = () => document.getElementById('new-key-modal').style.display = 'flex';
    window.hideCreateKeyModal = () => document.getElementById('new-key-modal').style.display = 'none';

    window.handleCreateKey = async function(e) {
        e.preventDefault();
        const name = document.getElementById('key-name').value;
        try {
            const result = await apiCall('/v1/keys', { method: 'POST', body: JSON.stringify({ name }) });
            hideCreateKeyModal();
            document.getElementById('new-key-value').textContent = result.api_key;
            document.getElementById('show-key-modal').style.display = 'flex';
            loadApiKeys();
        } catch (err) {
            alert('Failed to create key: ' + err.message);
        }
    };

    window.hideShowKeyModal = () => document.getElementById('show-key-modal').style.display = 'none';

    window.copyNewKey = function() {
        const key = document.getElementById('new-key-value').textContent;
        navigator.clipboard.writeText(key).then(() => alert('Copied!'));
    };

    window.revokeKey = async function(keyId) {
        if (!confirm('Are you sure you want to revoke this API key? This cannot be undone.')) return;
        try {
            await apiCall(`/v1/keys/${keyId}`, { method: 'DELETE' });
            loadApiKeys();
        } catch (e) {
            alert('Failed to revoke key: ' + e.message);
        }
    };

    // === Load Usage ===
    async function loadUsage() {
        try {
            const usage = await apiCall('/v1/billing/usage');
            const current = usage.current_usage || 0;
            const limit = usage.limit || 100;
            const pct = Math.min(100, Math.round((current / limit) * 100));

            document.getElementById('usage-current').textContent = current;
            document.getElementById('usage-limit').textContent = limit;
            document.getElementById('usage-pct').textContent = pct;
            const fill = document.getElementById('usage-fill');
            fill.style.width = pct + '%';
            fill.className = 'usage-fill' + (pct > 90 ? ' over-limit' : '');

            // Draw usage chart with mock history (replace with real data when available)
            drawUsageChart();
        } catch (e) {
            console.error('Failed to load usage:', e);
        }
    }

    function drawUsageChart() {
        const canvas = document.getElementById('usage-chart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        // Mock data - replace with actual historical usage
        const data = [20, 45, 30, 60, 25, 40];
        const months = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'];
        const max = Math.max(...data, 1);
        const barW = w / data.length - 20;
        const barGap = 20;

        ctx.fillStyle = '#e2e8f0';
        for (let i = 0; i < 5; i++) {
            const y = (h / 5) * i;
            ctx.fillRect(0, y, w, 1);
        }

        data.forEach((val, i) => {
            const barH = (val / max) * (h - 30);
            const x = i * (barW + barGap) + barGap / 2;
            const y = h - barH - 20;

            ctx.fillStyle = i === data.length - 1 ? '#0ea5e9' : '#cbd5e1';
            ctx.beginPath();
            ctx.roundRect(x, y, barW, barH, 4);
            ctx.fill();

            ctx.fillStyle = '#64748b';
            ctx.font = '11px Inter';
            ctx.textAlign = 'center';
            ctx.fillText(months[i], x + barW / 2, h - 5);
        });
    }

    // === Load Billing ===
    async function loadBilling() {
        try {
            const sub = await apiCall('/v1/billing/subscription');
            const usage = await apiCall('/v1/billing/usage');

            if (sub.has_subscription) {
                document.getElementById('billing-plan').textContent = (sub.tier || 'starter').charAt(0).toUpperCase() + (sub.tier || 'starter').slice(1);
                const prices = { starter: '$49/mo', professional: '$149/mo', enterprise: '$399/mo' };
                document.getElementById('billing-price').textContent = prices[sub.tier] || '$49/mo';
                document.getElementById('manage-billing-btn').style.display = 'inline-flex';
            } else {
                document.getElementById('billing-plan').textContent = 'Free Tier';
                document.getElementById('billing-price').textContent = '$0/mo';
            }
        } catch (e) {
            console.error('Failed to load billing:', e);
        }
    }

    window.openBillingPortal = async function() {
        try {
            const result = await apiCall('/v1/billing/portal', { method: 'POST' });
            if (result.portal_url) window.open(result.portal_url, '_blank');
        } catch (e) {
            alert('Failed to open billing portal: ' + e.message);
        }
    };

    // === Load Settings ===
    function loadSettings() {
        const user = getUser();
        document.getElementById('settings-name').value = user.name || '';
        document.getElementById('settings-email').value = user.email || '';
    }

    window.handleUpdateProfile = async function(e) {
        e.preventDefault();
        try {
            const user = await apiCall('/v1/me', {
                method: 'PUT',
                body: JSON.stringify({
                    name: document.getElementById('settings-name').value
                })
            });
            localStorage.setItem('user', JSON.stringify(user));
            document.getElementById('user-greeting').textContent = `Welcome, ${user.name}`;
            alert('Profile updated!');
        } catch (err) {
            alert('Failed to update profile: ' + err.message);
        }
    };

    window.handleChangePassword = async function(e) {
        e.preventDefault();
        try {
            await apiCall('/v1/auth/change-password', {
                method: 'POST',
                body: JSON.stringify({
                    current_password: document.getElementById('current-password').value,
                    new_password: document.getElementById('new-password').value
                })
            });
            alert('Password changed!');
            e.target.reset();
        } catch (err) {
            alert('Failed to change password: ' + err.message);
        }
    };

    window.handleDeleteAccount = async function() {
        if (!confirm('Are you sure? This will permanently delete your account and all data.')) return;
        if (!confirm('Really? This cannot be undone.')) return;
        try {
            await apiCall('/v1/me', { method: 'DELETE' });
            localStorage.clear();
            window.location.href = '/';
        } catch (e) {
            alert('Failed to delete account: ' + e.message);
        }
    };

    // === Logout ===
    window.handleLogout = function() {
        localStorage.clear();
        window.location.href = '/static/login.html';
    };

    // === Utility ===
    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // === Init ===
    document.addEventListener('DOMContentLoaded', function() {
        if (!checkAuth()) return;
        loadOverview();

        // Handle URL params for tab switching
        const params = new URLSearchParams(window.location.search);
        const tab = params.get('tab');
        if (tab) {
            const navEl = document.querySelector(`[data-tab="${tab}"]`);
            if (navEl) switchTab(tab, navEl);
        }
    });
})();
