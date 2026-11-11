"""Labelled prompts for the router.

STRUCTURED: PII the deterministic detectors must catch on their own, with the
detector expected. SEMANTIC: PII only Laya can see (no structured identifier).
NEGATIVE: must not trip any deterministic detector — heavy on dev/infra text,
because false positives there would route ordinary agent work to the small
local model.
"""

STRUCTURED = [
    ("Draft a reply to Sarah O'Connor, sarah.oconnor@gmail.com, about her overdue invoice.", "email"),
    ("My IBAN is IE29 AIBK 9311 5212 3456 78, can you set up the direct debit?", "iban"),
    ("Pay supplier DE89370400440532013000 by Friday.", "iban"),
    ("Credit card 4111 1111 1111 1111 exp 09/28 was declined, why?", "payment_card"),
    ("Card number 5555-5555-5555-4444 keeps failing at checkout.", "payment_card"),
    ("Ring my mum on 0851234567 and tell her I'll be late.", "phone"),
    ("Customer phone is +353 87 123 4567, call her back.", "phone"),
    ("Office line: (01) 555 0199, ask for accounts.", "phone"),
    ("Call 087 123 4567 on 2026-10-04 at 13:20 about the order.", "phone"),  # phone next to a date
    ("My colleague's phone is +353 86 123 4567. Repeat it back.", "phone"),  # before a full stop
    ("Ring 0851234567.", "phone"),
    ("His SSN is 123-45-6789, add it to the tax form.", "us_ssn"),
    ("National Insurance number AB 12 34 56 C for the payroll record.", "uk_nino"),  # QQ is HMRC's example prefix, invalid
    ("PPS number 1234567TW for the Revenue form.", "ie_ppsn"),
    ("Here is the passport number for the visa form: PA1234567, holder Liam Walsh.", "passport"),
    ("Patient Michael Byrne, DOB 14/03/1971, needs a follow-up.", "date_of_birth"),
    ("Born on March 3, 1985, she qualifies for the senior plan.", "date_of_birth"),
]

SEMANTIC = [
    "Please check why John Murphy's salary of 84,000 was paid late this month and email HR.",
    "Our employee Priya Nair has been on sick leave for depression since June; plan her return-to-work.",
    "I'm Dermot, I live at 4 Oak Lane in Cork and I keep getting letters for the previous tenant.",
    "Summarise the medical history of my client, Mrs. Eileen Doyle, who had a hip replacement in 2024.",
]

NEGATIVE = [
    "Write a Python function that validates email addresses with a regex.",
    "Explain the difference between TCP and UDP.",
    "What did Abraham Lincoln say in the Gettysburg Address?",
    "Summarise the GDPR rules on processing personal data.",
    "Refactor this function to use async/await: def fetch(url): return requests.get(url).json()",
    "Send the alert to ops@example.com and devs@example.org (placeholders for the docs).",
    "ssh -i ~/.ssh/id_ed25519 root@203.0.113.67 then check journalctl -u openshell-controller",
    "The gateway listens on 127.0.0.1:18789 and the controller on 0.0.0.0:3000.",
    "Epoch 1696339200 maps to 2023-10-03T12:00:00Z; convert 1727956800 too.",
    "Deployment id 7f345d95a060ab338d4cb44d749651e7139d53fa failed at step 3.",
    "Request id 3fa85f64-5717-4562-b3fc-2c963f66afa6 returned HTTP 502 after 30000 ms.",
    "Bump OpenShell from 0.0.106 to 0.0.116 and NemoClaw to v0.0.130.",
    "The table has 1234567 rows and 89012345 bytes of indexes.",
    "Set maxBuffer to 10485760 and timeout to 1200000 milliseconds.",
    "Order #100045672 shipped on 2026-10-03 from warehouse 12.",
    "Use sha256sum and compare with e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.",
    "Kubernetes pod web-7d4b9c6f5-x2kqz restarted 14 times.",
    "Version 2026.9.2 of OpenClaw fixes the gateway token race.",
    "Matrix dims are 4096 x 4096 with batch 32 and 8 heads.",
    "Port range 30000-32767 is reserved for NodePorts.",
    # OpenClaw 2026.9.x time context (live false positive, 2026-10-04)
    "Current time: Sunday, October 4th, 2026 - 1:20 PM (UTC)\nReference UTC: 2026-10-04 13:20 UTC\nExplain TCP vs UDP.",
    "Timestamp 2026-10-04T13:20:45.123Z, retried at 2026-10-04 13:21:02.",
    "The job ran on 04/10/2026 at 13:20 and again on 05/10/2026 09:15.",
]
