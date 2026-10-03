import os
import json
import requests
import html
import urllib.parse
from typing import Optional, Any, List
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
WEBSITE_URL = os.getenv("FRONTEND_APP_URL", "https://politymentor.com").rstrip("/")


def clean(text: Any) -> str:
    """Safely escapes HTML entities."""
    return html.escape(str(text or "").strip())


class TelegramSmartNotesNotifier:
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_capsule_message(
        self,
        study_intelligence: Any,
        chapter: str,
        topic: str,
        prid: Any,
        object_key: str
    ) -> bool:
        """Sends the structured synthesis post with the FlutterFlow CurrentAffairs4 deep link."""
        if not self.is_configured():
            print("  [TELEGRAM] ⚠️ Bot credentials not configured in .env. Skipping notification.")
            return False

        data = study_intelligence.model_dump(mode="json") if hasattr(study_intelligence, "model_dump") else study_intelligence

        headline = clean(data.get("headline", topic))
        summary = clean(data.get("summary", ""))
        why_it_matters = clean(data.get("why_it_matters", ""))
        mains_q = clean(data.get("mains_question", ""))

        parts = [
            f"🏛 <b>UPSC GS-II DAILY INTELLIGENCE CAPSULE</b>",
            f"📖 <b>Chapter:</b> {clean(chapter)} | <b>Topic:</b> {clean(topic)}",
            f"🆔 <b>PIB Reference:</b> <code>PRID {prid}</code>\n",
            f"📌 <b>{headline}</b>\n",
            f"⚡ <b>EXECUTIVE SUMMARY:</b>\n{summary}\n"
        ]

        # 4 core takeaways
        smart_notes: List[str] = data.get("smart_notes") or []
        if smart_notes:
            parts.append("📊 <b>KEY ANALYSIS & FACTUAL TAKEAWAYS:</b>")
            for note in smart_notes[:4]:
                parts.append(f"• {clean(note)}")
            parts.append("")

        # Static constitutional linkages
        linkages: List[str] = data.get("backward_linkages") or []
        if linkages:
            parts.append("🔗 <b>BACKWARD LINKAGES (CONSTITUTION & POLITY):</b>")
            for link in linkages[:4]:
                parts.append(f"▫️ {clean(link)}")
            parts.append("")

        # Mains Dimension
        if mains_q:
            parts.append(f"✍️ <b>MAINS ANSWER WRITING FRAMEWORK:</b>\n<i>\"{mains_q}\"</i>\n")

        # Encode object key for deep-linking
        encoded_key = urllib.parse.quote(object_key, safe="")
        deep_link = f"{WEBSITE_URL}/CurrentAffairs4?object_key={encoded_key}"

        cta_banner = (
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "💡 <b>Deepen Your Preparation with AI:</b>\n"
            "Interrogate contradictory constitutional provisions, generate model answers, and draft personal notes inside your workspace.\n\n"
            f"👉 <b>Open Interactive Note:</b> <a href=\"{deep_link}\">Read Full Smart Note</a>"
        )
        parts.append(cta_banner)

        full_message = "\n".join(parts)

        # Telegram character guard
        if len(full_message) > 4000:
            full_message = full_message[:3880] + f"...\n\n👉 <b>Continue on</b> <a href=\"{deep_link}\">politymentor.com</a>"

        payload = {
            "chat_id": self.chat_id,
            "text": full_message,
            "parse_mode": "HTML",
            "disable_web_page_preview": False
        }

        try:
            resp = requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=12)

            if not resp.ok:
                print(f"  [TELEGRAM ERROR] sendMessage failed: {resp.status_code} - {resp.text}")
                return False
            return True
        except Exception as e:
            print(f"  [TELEGRAM ERROR] Message dispatch exception: {e}")
            return False

    def send_prelims_quiz(self, study_intelligence: Any, prid: Any) -> bool:
        """Publishes an interactive Quiz Poll using the first prelims practice item."""
        if not self.is_configured():
            return False

        data = study_intelligence.model_dump(mode="json") if hasattr(study_intelligence, "model_dump") else study_intelligence
        prelims_items = data.get("prelims_practice_points") or []
        if not prelims_items:
            print("  [TELEGRAM] No prelims practice points available for Quiz Poll.")
            return False

        q_item = prelims_items[0]
        statement = str(q_item.get("statement", "")).strip()
        is_correct = bool(q_item.get("is_correct"))
        raw_explanation = str(q_item.get("explanation", "")).strip()

        # Constraints imposed by Telegram Poll API
        poll_question = f"🎯 [Prelims Check • PRID {prid}]\n{statement}"
        if len(poll_question) > 300:
            poll_question = poll_question[:295] + "..."

        explanation = raw_explanation[:195] + "..." if len(raw_explanation) > 195 else raw_explanation

        options = ["Statement is Correct", "Statement is Incorrect"]
        correct_option_id = 0 if is_correct else 1

        payload = {
            "chat_id": self.chat_id,
            "question": poll_question,
            "options": json.dumps(options),
            "is_anonymous": True,
            "type": "quiz",
            "correct_option_id": correct_option_id,
            "explanation": explanation
        }

        try:
            resp = requests.post(f"{self.base_url}/sendPoll", data=payload, timeout=10)
            if not resp.ok:
                print(f"  [TELEGRAM ERROR] sendPoll failed: {resp.status_code} - {resp.text}")
                return False
            return True
        except Exception as e:
            print(f"  [TELEGRAM ERROR] Poll dispatch exception: {e}")
            return False

    def broadcast_note_and_quiz(
        self,
        study_intelligence: Any,
        chapter: str,
        topic: str,
        prid: Any,
        object_key: str
    ) -> bool:
        """Sends both the synthesis card and the interactive quiz sequentially."""
        msg_ok = self.send_capsule_message(study_intelligence, chapter, topic, prid, object_key)
        poll_ok = self.send_prelims_quiz(study_intelligence, prid)
        return msg_ok and poll_ok
