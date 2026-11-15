# FinVision Firebase deployment checklist

This project uses the Firebase Admin SDK from the Streamlit server. The Admin SDK bypasses Firestore and Storage Security Rules, so protect the Streamlit app and its server-side credentials carefully. Never put service-account credentials in browser code or commit them to Git.

## 1. Prepare the Firebase project

In the Firebase console, select the Firebase project intended for FinVision:

1. Enable **Authentication → Sign-in method → Email/Password**.
2. Create a **Cloud Firestore** database.
3. Create a **Cloud Storage** bucket.
4. In **Project settings → General**, copy the Web API key.
5. In **Project settings**, note the exact project ID and Storage bucket name.

Do not assume the bucket ends in `.appspot.com`; use the exact bucket name shown in your Firebase project.

## 2. Create server credentials

For a Streamlit-hosted app, create a dedicated service account with only the permissions the app needs for Firebase Authentication user administration, Firestore data access, and Cloud Storage object access. Download its JSON key only if your hosting platform cannot use workload identity.

Treat the JSON key as a password:
- Never commit it to GitHub.
- Never place it in a public repository, issue, screenshot, or frontend bundle.
- If it was ever committed, revoke/delete that key and create a new one.

## 3. Configure Streamlit Community Cloud secrets

Open your Streamlit app settings → **Secrets** and add the following TOML values. Replace every placeholder with values from your own Firebase project.

```toml
FIREBASE_USE_EMULATOR = "false"
FIREBASE_PROJECT_ID = "YOUR_FIREBASE_PROJECT_ID"
FIREBASE_STORAGE_BUCKET = "YOUR_EXACT_STORAGE_BUCKET_NAME"
FIREBASE_API_KEY = "YOUR_FIREBASE_WEB_API_KEY"

FIREBASE_SERVICE_ACCOUNT_JSON = '''
{"type":"service_account","project_id":"YOUR_FIREBASE_PROJECT_ID","private_key_id":"REPLACE_ME","private_key":"-----BEGIN PRIVATE KEY-----\\nREPLACE_ME\\n-----END PRIVATE KEY-----\\n","client_email":"REPLACE_ME","client_id":"REPLACE_ME","token_uri":"https://oauth2.googleapis.com/token"}
'''
```

Paste the **complete** service-account JSON content into the triple-quoted value. Keep it private. Do not use the example JSON above as a real credential.

For local development, use a private `.env` file (which must remain gitignored) and set `FIREBASE_USE_EMULATOR=true` only when you are intentionally running the Firebase emulators.

## 4. Deploy security rules

From the repository directory, after installing and logging into the Firebase CLI:

```powershell
firebase login
firebase use --add
firebase deploy --only firestore:rules,storage
```

Choose the same Firebase project configured in Streamlit secrets. Review the rules against the actual application data model before relying on them. The server-side Admin SDK bypasses these rules; they protect direct client SDK access.

## 5. Verify before inviting judges

- Confirm the Streamlit app starts with emulator mode disabled.
- Create a normal analyst account and sign in.
- Upload a harmless test invoice.
- Confirm its document record appears in Firestore.
- Confirm the original file exists in the configured Storage bucket.
- Restart/redeploy the Streamlit app and confirm history still exists.
- Confirm a newly registered account cannot obtain the admin role.
- Confirm a user cannot access another user's documents.
- Check Firebase Authentication, Firestore and Storage usage/billing limits.

**Important:** This configuration makes production failures visible instead of silently pretending that temporary in-memory data was saved. It does not by itself constitute a security audit or make the application compliant for real financial/customer data.
