from slack_sdk import WebClient


def is_admin(user_id: str, client: WebClient) -> bool:
    try:
        user_info = client.users_info(user=user_id)
        is_admin = user_info["user"]["is_admin"] or user_info["user"]["is_owner"]
        return is_admin
    except Exception as e:
        print(f"❌ Failed to check user permissions: {e}")
        return False
