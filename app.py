def send_results_email(to_email: str, subject: str, body: str,
                       attachment_bytes: bytes, filename: str):
    import smtplib
    from email.message import EmailMessage

    host = st.secrets["smtp"]["host"]
    port = int(st.secrets["smtp"].get("port", 587))
    user = st.secrets["smtp"]["user"]
    password = st.secrets["smtp"]["password"]
    use_tls = bool(st.secrets["smtp"].get("use_tls", True))

    # Prefer explicit from_email; otherwise use the authenticated user
    from_email = st.secrets["smtp"].get("from_email", user)

    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    msg.add_attachment(
        attachment_bytes,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename
    )

    # Fresh SMTP connection each time (prevents "connect() first")
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        if use_tls:
            smtp.starttls()
            smtp.ehlo()
        smtp.login(user, password)
        smtp.send_message(msg)
