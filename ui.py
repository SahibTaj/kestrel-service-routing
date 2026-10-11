import streamlit as st
import requests


st.set_page_config(
    page_title="Kestrel Routing Desk",
    page_icon="🧭",
    layout="centered",
)
st.title("Kestrel Service Request Router")
st.caption("Local ML baseline · always review ambiguous cases before production routing")

with st.form("request"):
    request_id = st.text_input("Request ID (optional)")
    channel = st.selectbox("Channel", ["chat", "whatsapp", "email", "ivr"])
    product = st.selectbox(
        "Product family",
        [
            "Water Purifier",
            "Air Fryer",
            "Mixer Grinder",
            "Robot Vacuum",
            "Cooktop",
            "Fan",
            "Heater",
        ],
    )
    warranty = st.selectbox(
        "Warranty status",
        ["in_warranty", "shield", "out_of_warranty"],
    )
    source = st.selectbox("Source", ["crm", "legacy_zoho"])
    body = st.text_area(
        "Customer message",
        placeholder="Describe the customer's request...",
        height=140,
    )
    submitted = st.form_submit_button("Route request", type="primary")

if submitted:
    if not body.strip():
        st.error("Enter a customer message first.")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={
                    "request_id": request_id or None,
                    "channel": channel,
                    "product_family": product,
                    "warranty_status": warranty,
                    "source": source,
                    "request_text": body,
                },
                timeout=10,
            )

            if response.ok:
                result = response.json()
                st.success(f"Suggested team: {result['team']}")

                if result.get("review_required"):
                    st.warning("Policy review recommended before routing")

                st.subheader("Why this team?")
                for reason in result["reasons"]:
                    st.write("- " + reason)

                st.caption(result.get("note", ""))
            else:
                st.error(
                    f"Service returned {response.status_code}: {response.text}"
                )
        except requests.RequestException:
            st.error(
                "Could not reach the API. Start it first with: "
                "uvicorn app:app --reload"
            )
