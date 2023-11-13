
import smtplib
from email.message import EmailMessage


def send_verification_email(recipient_email: str, token: str):
    try:  
        # Create an EmailMessage
        message = EmailMessage()
        message.set_content(f"Your verification token is: {token}")

        # Set sender and recipient email addresses
        message['From'] = 'sfrhinoai@gmail.com'  # Replace with your sender email 

        message['To'] = recipient_email

        # Set the subject
        message['Subject'] = 'Verification Token'

        # Connect to Gmail's SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Start TLS encryption

        # Log in to your Gmail account
        server.login('sfrhinoai@gmail.com', 'jhvo lgao pmfb cvek')  # Replace with your actual email and password

        # Send the email
        server.send_message(message)

        # Close the SMTP server connection
        server.quit()

        return True  # Email sent successfully
    except Exception as e:
        print(f"An error occurred while sending email: {e}")
        return False  # Email sending failed


def send_email(recipient_email: str, subject: str, body: str):
    try:
        # Create an EmailMessage
        message = EmailMessage()
        message.set_content(body)

        # Set sender and recipient email addresses
        message['From'] = 'sfrhinoai@gmail.com'  # Replace with your sender email
        message['To'] = recipient_email

        # Set the subject
        message['Subject'] = subject

        # Connect to the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()

        # Log in to the email account
        server.login('sfrhinoai@gmail.com', 'jhvo lgao pmfb cvek')  # Replace with your actual email and password

        # Send the email
        server.send_message(message)

        # Close the server
        server.quit()

        return True
    except Exception as e:
        print(f"An error occurred while sending email: {e}")
        return False
    