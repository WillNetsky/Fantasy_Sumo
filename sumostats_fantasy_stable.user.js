// ==UserScript==
// @name         SumoStats Fantasy Stable Highlighter & Notifier
// @namespace    http://tampermonkey.net/
// @version      2.1
// @description  Highlights your fantasy rikishi stable members and provides notifications for upcoming or active bouts on sumostats.com. Reacts to dynamic content changes.
// @author       Gemini Code Assist
// @match        *://sumostats.com/live/*
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_addStyle
// @grant        GM_notification
// @run-at       document-idle
// ==/UserScript==

(function() {
    'use strict';

    // --- CONFIGURATION ---
    const STABLE_STORAGE_KEY = 'sumostats_fantasy_stable';
    const HIGHLIGHT_CLASS = 'fantasy-stable-rikishi';
    const NEXT_BOUT_CLASS = 'next-bout-notification';

    // CSS for the script
    const STYLE_CSS = `
        /* Styles for highlighted rikishi */
        .${HIGHLIGHT_CLASS} {
            font-weight: 700 !important;
            color: #10B981 !important; /* Tailwind green-500 */
            background-color: #ECFDF5 !important; /* Light green background */
            border-bottom: 2px solid #059669;
            padding: 2px 4px;
            border-radius: 4px;
            display: inline-block;
            line-height: 1.2;
            transition: all 0.3s ease;
        }

        /* Styles for the config panel */
        #stable-config-panel {
            position: fixed;
            top: 20px;
            right: 20px;
            width: 300px;
            background-color: #fff;
            border: 1px solid #ccc;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            padding: 15px;
            z-index: 10000;
            font-family: sans-serif;
            color: #333;
        }
        #stable-config-panel h3 {
            margin-top: 0;
            color: #333;
            font-size: 1.1em;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
        }
        #stable-config-panel textarea {
            width: 100%;
            height: 100px;
            border: 1px solid #ddd;
            padding: 8px;
            margin-bottom: 10px;
            box-sizing: border-box;
            resize: vertical;
            font-size: 14px;
            line-height: 1.4;
            color: #333;
        }
        #stable-config-panel button {
            background-color: #3B82F6; /* Tailwind blue-500 */
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
            transition: background-color 0.2s;
        }
        #stable-config-panel button:hover {
            background-color: #2563EB; /* Tailwind blue-600 */
        }
        #stable-config-panel .close-button {
            position: absolute;
            top: 5px;
            right: 10px;
            background: none;
            color: #888;
            font-size: 1.2em;
        }

        /* Notification for next bout */
        .${NEXT_BOUT_CLASS}::after {
            content: " 🚨 NEXT";
            color: white;
            background-color: #EF4444; /* Tailwind red-500 */
            padding: 1px 5px;
            margin-left: 5px;
            border-radius: 9999px; /* Pill shape */
            font-size: 0.7em;
            vertical-align: middle;
            font-weight: 700;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.8; }
        }
    `;

    let stableMembers = [];
    let notifiedBoutText = null; // State to prevent re-notifying for the same bout

    // --- UTILITIES ---

    /**
     * Debounce function to limit how often a function can run.
     */
    function debounce(func, delay) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), delay);
        };
    }

    // --- STORAGE UTILITIES ---

    function loadStable() {
        const storedValue = GM_getValue(STABLE_STORAGE_KEY, '');
        stableMembers = storedValue ? storedValue.split(',').map(name => name.trim().toLowerCase()).filter(Boolean) : [];
        return stableMembers;
    }

    function saveStable(names) {
        const valueToStore = names.map(name => name.trim()).filter(Boolean).join(',');
        GM_setValue(STABLE_STORAGE_KEY, valueToStore);
        loadStable();
        scanEntirePage(); // Use the main scan function
        showSimpleMessage('Stable updated successfully!');
    }

    // --- UI/MESSAGING UTILITIES ---

    function showSimpleMessage(message) {
        const msgDiv = document.createElement('div');
        msgDiv.id = 'gm-user-message';
        GM_addStyle(`
            #gm-user-message {
                position: fixed; bottom: 20px; left: 50%;
                transform: translateX(-50%); background-color: #333; color: white;
                padding: 10px 20px; border-radius: 5px; z-index: 10001;
                opacity: 0; transition: opacity 0.5s ease-in-out;
            }
        `);
        msgDiv.textContent = message;
        document.body.appendChild(msgDiv);
        setTimeout(() => { msgDiv.style.opacity = '1'; }, 50);
        setTimeout(() => {
            msgDiv.style.opacity = '0';
            setTimeout(() => msgDiv.remove(), 500);
        }, 3000);
    }

    function createConfigUI() {
        if (document.getElementById('stable-config-panel')) return;
        const panel = document.createElement('div');
        panel.id = 'stable-config-panel';
        panel.innerHTML = `
            <h3>Fantasy Stable Management</h3>
            <p style="font-size: 0.8em; margin-bottom: 10px;">Enter rikishi names, one per line or separated by commas.</p>
            <textarea id="rikishi-stable-input" placeholder="e.g., Hakuho, Terunofuji, Takerufuji">${stableMembers.join('\n')}</textarea>
            <button id="save-stable-btn">Save Stable</button>
            <button class="close-button" id="close-panel-btn">×</button>
        `;
        document.body.appendChild(panel);

        const inputEl = document.getElementById('rikishi-stable-input');
        const saveBtn = document.getElementById('save-stable-btn');
        const closeBtn = document.getElementById('close-panel-btn');

        saveBtn.addEventListener('click', () => {
            const newNames = inputEl.value.split(/[\n,]/).map(name => name.trim()).filter(Boolean);
            saveStable(newNames);
            panel.remove();
        });
        closeBtn.addEventListener('click', () => panel.remove());
    }

    // --- CORE LOGIC ---

    /**
     * Clears all highlights and performs a full, two-pass scan of the document
     * to correctly identify the single "next" bout.
     */
    function scanEntirePage() {
        // 1. Clear all existing highlights and notification tags
        document.querySelectorAll(`.${HIGHLIGHT_CLASS}`).forEach(el => {
            el.classList.remove(HIGHLIGHT_CLASS, NEXT_BOUT_CLASS);
        });

        if (stableMembers.length === 0) return;

        // 2. PASS 1: Scan, highlight, and collect candidates
        const potentialNextBouts = [];
        const elements = document.body.querySelectorAll('a, span, td, p, div');

        elements.forEach(el => {
            const text = el.textContent.trim();
            if (!text) return;

            if (stableMembers.includes(text.toLowerCase())) {
                el.classList.add(HIGHLIGHT_CLASS);

                const boutContainer = el.closest('div[style*="grid-template-columns"]');
                if (boutContainer) {
                    // A match has a result if the row contains a "Win" icon. This is the most reliable indicator.
                    const winIndicator = boutContainer.querySelector('span[title="Win"]');
                    const hasResult = !!winIndicator; // Convert found element or null to a true/false boolean.

                    const isLive = !!boutContainer.querySelector('span[title="Current live bout"]');

                    if (!hasResult || isLive) {
                        potentialNextBouts.push({ element: el, isLive: isLive, text: text });
                    }
                }
            }
        });

        // 3. PASS 2: Decide which candidate is the correct "next" one
        if (potentialNextBouts.length > 0) {
            const nextBout = potentialNextBouts[potentialNextBouts.length - 1];
            nextBout.element.classList.add(NEXT_BOUT_CLASS);

            // --- NOTIFICATION LOGIC ---
            // Only trigger a notification if the bout is LIVE and we haven't notified for this wrestler yet.
            if (nextBout.isLive && notifiedBoutText !== nextBout.text) {
                GM_notification({
                    title: "Fantasy Stable Alert!",
                    text: `${nextBout.text}'s bout is LIVE!`,
                    silent: false, // Set to false to play a sound
                    timeout: 10000
                });
                notifiedBoutText = nextBout.text;
            }
        } else {
            // No upcoming bouts found, so reset the notification tracker
            notifiedBoutText = null;
        }
    }

    // --- INITIALIZATION ---

    function init() {
        GM_addStyle(STYLE_CSS);
        loadStable();
        scanEntirePage(); // Perform initial scan

        // Set up the config button
        const configBtn = document.createElement('button');
        configBtn.id = 'stable-config-toggle';
        GM_addStyle(`#stable-config-toggle {
            position: fixed; bottom: 20px; right: 20px; background-color: #6366F1;
            color: white; padding: 10px 15px; border: none; border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); cursor: pointer;
            z-index: 9999; font-size: 14px; font-weight: 700;
        }`);
        configBtn.textContent = 'Manage Stable';
        configBtn.addEventListener('click', createConfigUI);
        document.body.appendChild(configBtn);

        // Set up the MutationObserver with a debounced rescan for efficiency
        const debouncedScan = debounce(scanEntirePage, 500);
        const observer = new MutationObserver(() => {
            debouncedScan();
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    init();

})();
