import os
from dotenv import load_dotenv
from src.storage.minio_client import MinIOStorage
from src.services.telegram_notifier import send_full_smart_note_post

load_dotenv()

def run_telegram_test():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    print(f"Token configured: {'Yes' if bot_token else 'No'}")
    print(f"Target Chat ID: {chat_id}")

    if not (bot_token and chat_id):
        print("❌ Please define TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env first.")
        return

    # Fetch 1 existing note from storage to test with real data
    storage = MinIOStorage()
    notes = storage.list_latest_smart_notes(limit=1)

    if not notes:
        print("No notes found in storage. Using dummy data for test.")
        sample_data = {
            "headline": "Empowering Grassroots Institutions: National Panchayat Awards",
            "summary": "The Ministry of Panchayati Raj reviewed rural decentralization initiatives emphasizing direct benefit transfers and digital accountability.",
            "why_it_matters": "Directly links to Article 243G, Eleventh Schedule subjects, and fiscal federalism at local government tiers.",
            "prelims_practice_points": [
                "73rd Constitutional Amendment Act, 1992 added Part IX.",
                "Article 243-I mandates State Finance Commission every 5 years.",
                "Gram Sabha is the cornerstone of participatory democracy."
            ],
            "mains_question": "Examine the functional and financial challenges faced by Panchayati Raj Institutions in realizing genuine grassroots decentralization."
        }
        chapter = "Panchayati Raj"
        topic = "Local Governance"
        prid = 2318441
    else:
        latest = notes[0]
        sample_data = latest.get("smart_notes", {})
        chapter = latest.get("chapter", "Polity")
        topic = latest.get("topic", "Governance")
        prid = latest.get("prid", 0)

    print(f"\nDispatching test broadcast for PRID: {prid} ...")
    success = send_full_smart_note_post(
        study_intelligence_data=sample_data,
        chapter=chapter,
        topic=topic,
        prid=prid
    )

    if success:
        print("✅ Telegram notification delivered successfully! Check your channel/chat.")
    else:
        print("❌ Delivery failed. Check terminal error logs above.")

if __name__ == "__main__":
    run_telegram_test()
