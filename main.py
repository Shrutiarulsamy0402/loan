

import streamlit as st
import pandas as pd
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_loan_email(to_email, user_name, loan_id, status, remarks):
    from_email = st.secrets["EMAIL_ADDRESS"]
    from_password = st.secrets["EMAIL_PASSWORD"]

    # ✅ Your Indian Bank logo hosted on imgbb
    logo_url = "https://companieslogo.com/img/orig/INDIANB.NS_BIG-f675f730.png?t=1615846835"


    subject = f"Indian Bank - Your Loan Application is {status.capitalize()} (Loan ID: {loan_id})"

    if status == "approved":
        greeting = "🎉 <strong>Congratulations!</strong>"
        msg_line = "We are pleased to inform you that your loan application has been <strong>approved</strong>."
    elif status == "declined":
        greeting = "<strong>Thank you for applying.</strong>"
        msg_line = "After careful review, we regret to inform you that your loan application has been <strong>declined</strong>."
    else:
        greeting = "<strong>Application Received!</strong>"
        msg_line = "Your loan application is currently <strong>under review</strong>. We’ll notify you once a decision is made."

    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; padding: 20px;">
        <div style="max-width: 600px; margin: auto; border: 1px solid #e0e0e0; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
          <div style="background-color: #003366; padding: 10px 20px; border-top-left-radius: 10px; border-top-right-radius: 10px; text-align: center;">
            <img src="{logo_url}" alt="Indian Bank Logo" style="height: 60px;" />
          </div>
          <div style="padding: 20px;">
            <p>Dear {user_name},</p>
            <p>{greeting}</p>
            <p>{msg_line}</p>
            <table style="margin-top: 10px; font-size: 15px;">
              <tr><td><strong>📝 Loan ID:</strong></td><td>{loan_id}</td></tr>
              <tr><td><strong>📅 Status:</strong></td><td>{status.capitalize()}</td></tr>
              <tr><td><strong>💬 Remarks:</strong></td><td>{remarks}</td></tr>
            </table>
            <p style="margin-top: 20px;">Thank you for choosing <strong>Indian Bank</strong>.</p>
            <p>We're committed to supporting your financial goals.</p>
            <p style="margin-top: 30px;">Warm regards,<br><strong>Indian Bank Loan Department</strong></p>
          </div>
        </div>
      </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_email, from_password)
            server.send_message(msg)
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")



    # Set up email
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))



    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(from_email, from_password)
            server.send_message(msg)
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# Paths to CSV files
data_path = "data"
users_file = os.path.join(data_path, "users.csv")
accounts_file = os.path.join(data_path, "accounts.csv")
loans_file = os.path.join(data_path, "loan_applications.csv")
loan_status_file = os.path.join(data_path, "loan_status.csv")
transactions_file = os.path.join(data_path, "transactions.csv")

# Load and Save CSV
def load_csv(file, expected_columns=None):
    try:
        if os.path.exists(file):
            df = pd.read_csv(file)
            if expected_columns:
                for col in expected_columns:
                    if col not in df.columns:
                        df[col] = np.nan
            return df
        else:
            return pd.DataFrame(columns=expected_columns if expected_columns else [])
    except Exception as e:
        st.error(f"Error loading {file}: {e}")
        return pd.DataFrame(columns=expected_columns if expected_columns else [])

def save_csv(df, file):
    try:
        df.to_csv(file, index=False)
    except Exception as e:
        st.error(f"Error saving {file}: {e}")

# Load data into session state
def load_data_to_session():
    st.session_state.users_df = load_csv(users_file, ["user_id", "username", "password", "role"])
    st.session_state.accounts_df = load_csv(accounts_file, ["user_id", "account_no", "address", "mobile", "balance"])
    st.session_state.loans_df = load_csv(loans_file, ["loan_id", "user_id", "amount", "purpose", "income", "status", "application_date", "remarks"])
    st.session_state.loan_status_df = load_csv(loan_status_file, ["loan_id", "user_id", "amount", "purpose", "income", "status", "application_date", "remarks"])
    st.session_state.transactions_df = load_csv(transactions_file, ["user_id", "loan_id", "amount", "method", "date"])

load_data_to_session()

users_df = st.session_state.users_df
accounts_df = st.session_state.accounts_df
loans_df = st.session_state.loans_df
loan_status_df = st.session_state.loan_status_df
transactions_df = st.session_state.transactions_df

if "user" not in st.session_state:
    st.session_state.user = None

# Ensure required columns exist
def ensure_columns():
    global users_df, accounts_df, loans_df

    if 'status' not in loans_df.columns:
        loans_df['status'] = 'pending'
    if 'remarks' not in loans_df.columns:
        loans_df['remarks'] = ''
    for col in ['account_no', 'address', 'balance', 'mobile']:
        if col not in accounts_df.columns:
            accounts_df[col] = '' if col != 'balance' else 0
    for col in ['username', 'user_id', 'password', 'role']:
        if col not in users_df.columns:
            users_df[col] = ''

ensure_columns()

if "user" not in st.session_state:
    st.session_state.user = None

# User Registration
# Create New User
def create_new_user():
    st.title("Create New User Account")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    role = st.selectbox("Role", ["user"])
    city = st.text_input("City")
    mobile = st.text_input("Mobile Number (e.g., xxxxxxx237)")

    if st.button("Create Account"):
        if username in users_df["username"].values:
            st.error("Username already exists. Please choose another.")
        else:
            user_id = f"U{len(users_df)+1:04d}"
            new_user = pd.DataFrame([{"user_id": user_id, "username": username, "password": password, "role": role}])
            new_account = pd.DataFrame([{"user_id": user_id, "account_no": f"XXXXXXX{random.randint(100,999)}", "address": city, "mobile": mobile, "balance": 0}])

            updated_users = pd.concat([users_df, new_user], ignore_index=True)
            updated_accounts = pd.concat([accounts_df, new_account], ignore_index=True)

            save_csv(updated_users, users_file)
            save_csv(updated_accounts, accounts_file)

            st.success("Account created successfully!")
# Login Function
def login():
    

    # ✅ Add Bank Logo
    st.markdown("""
        <div style="text-align: center;">
            <img src="https://imgs.search.brave.com/3snmeE1h6X_V2LAWSpZoHYAuqzMDjpb1t-6h-_oV_4I/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZG4u/ZnJlZWxvZ292ZWN0/b3JzLm5ldC93cC1j/b250ZW50L3VwbG9h/ZHMvMjAxOS8wMi9p/bmRpYW4tYmFuay1s/b2dvLnBuZw" width="350">
        </div>
    """, unsafe_allow_html=True)

    menu = st.radio("Select an option", ["Login", "Create Account", "Forgot Password?"])

    if menu == "Create Account":
        create_new_user()
        return

    if menu == "Forgot Password?":
        st.subheader("Reset Your Password with Mobile Verification")
        username = st.text_input("Enter your username")
        mobile = st.text_input("Enter your registered mobile number")
        new_password = st.text_input("Enter your new password", type="password")

        if st.button("Reset Password"):
            users_df = load_csv(users_file)
            accounts_df = load_csv(accounts_file)

            user_row = users_df[users_df["username"] == username]
            if user_row.empty:
                st.error("❌ Username not found.")
                return

            user_id = user_row.iloc[0]["user_id"]
            acc_row = accounts_df[(accounts_df["user_id"] == user_id) & (accounts_df["mobile"] == mobile)]

            if acc_row.empty:
                st.error("❌ Mobile number does not match our records.")
            else:
                users_df.loc[users_df["username"] == username, "password"] = hash_password(new_password)
                save_csv(users_df, users_file)
                st.success("✅ Password reset successful! You may now log in.")
        return

    # Login form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users_df = load_csv(users_file)

        required_cols = {"username", "password", "role", "user_id"}
        if not required_cols.issubset(set(users_df.columns)):
            st.error("Error: 'users.csv' is missing required columns.")
            st.stop()

        user = users_df[
            (users_df["username"] == username) &
            (users_df["password"] == password)
        ]

        if not user.empty:
            st.session_state.user = user.iloc[0].to_dict()
            st.success(f"Logged in as {username}")
            st.rerun()
        else:
            st.error("Invalid username or password")



# Admin Dashboard
def admin_dashboard():
    st.sidebar.title("Admin Panel")
    option = st.sidebar.radio("Select", [
        "📃 All Applications",
        "✅ Pending Loans",
        "🔍 Fetch User Info",
        "📊 Loan Summary & Analytics"
    ])

    if option == "📃 All Applications":
        st.subheader("All Loan Applications")
        sort_option = st.selectbox("🔍 Filter by Loan Status", ["All", "approved", "pending", "declined"])
        filtered_loans = loans_df if sort_option == "All" else loans_df[loans_df["status"] == sort_option]
        st.dataframe(filtered_loans.reset_index(drop=True))

    elif option == "✅ Pending Loans":
        st.subheader("Pending Loans (Average Risk)")

        train_df = loans_df[loans_df["status"] != "pending"]
        if train_df.empty or len(train_df["status"].unique()) < 2:
            st.warning("Not enough historical data to train model.")
            return

        train_df = train_df[["amount", "income", "status"]].dropna()
        X = train_df[["amount", "income"]]
        y = (train_df["status"] == "approved").astype(int)

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)

        pending_loans = loans_df[loans_df["status"] == "pending"]
        if pending_loans.empty:
            st.info("No pending loans to review.")
            return

        review_required = []
        for idx, row in pending_loans.iterrows():
            X_test = np.array([[row["amount"], row["income"]]])
            prob = model.predict_proba(X_test)[0][1]
            risk_score = round((1 - prob) * 100, 2)
            loan_id = row["loan_id"]
            remark = f"Predicted Risk Score: {risk_score}%"

            if risk_score <= 39:
                loans_df.loc[loans_df["loan_id"] == loan_id, "status"] = "approved"
                loans_df.loc[loans_df["loan_id"] == loan_id, "remarks"] = f"Auto-approved. {remark}"
                loan_status_df.loc[loan_status_df["loan_id"] == loan_id, "status"] = "approved"
                loan_status_df.loc[loan_status_df["loan_id"] == loan_id, "remarks"] = f"Auto-approved. {remark}"
                st.success(f"✅ Loan {loan_id} auto-approved (Low Risk)")

            elif risk_score >= 61:
                auto_reason = random.choice([
                    "Low credit score based on prior history",
                    "Insufficient income compared to requested amount",
                    "Debt-to-income ratio too high",
                    "Missing financial documentation",
                    "No verifiable employment details",
                    "Loan amount exceeds eligibility",
                    "Unstable job profile detected",
                    "Repeated rejections in past applications",
                    "Incomplete KYC compliance",
                    "Application failed risk assessment rules"
                ])
                full_remark = f"Auto-declined: {auto_reason}. {remark}"
                loans_df.loc[loans_df["loan_id"] == loan_id, "status"] = "declined"
                loans_df.loc[loans_df["loan_id"] == loan_id, "remarks"] = full_remark
                loan_status_df.loc[loan_status_df["loan_id"] == loan_id, "status"] = "declined"
                loan_status_df.loc[loan_status_df["loan_id"] == loan_id, "remarks"] = full_remark
                st.error(f"❌ Loan {loan_id} auto-declined (High Risk)\n📝 Reason: {auto_reason}")
            else:
                row["risk_score"] = risk_score
                review_required.append(row)

        save_csv(loans_df, loans_file)
        save_csv(loan_status_df, loan_status_file)

        if review_required:
            st.warning("⚠️ Loans requiring admin review (Average Risk)")

            for row in review_required:
                loan_id = row["loan_id"]
                risk_score = row["risk_score"]
                st.write(f"### Loan ID: {loan_id}")
                st.write(row.drop("risk_score"))
                st.info(f"Predicted Risk Score: {risk_score}%")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"✅ Approve {loan_id}", key=f"approve_{loan_id}"):
                        loans_df.loc[loans_df["loan_id"] == loan_id, "status"] = "approved"
                        loans_df.loc[loans_df["loan_id"] == loan_id, "remarks"] = f"Admin-approved. Risk Score: {risk_score}%"
                        loan_status_df.loc[loan_status_df["loan_id"] == loan_id, "status"] = "approved"
                        loan_status_df.loc[loan_status_df["loan_id"] == loan_id, "remarks"] = f"Admin-approved. Risk Score: {risk_score}%"
                        save_csv(loans_df, loans_file)
                        save_csv(loan_status_df, loan_status_file)
                        st.success(f"Loan {loan_id} approved")
                        st.rerun()

                with col2:
                    with st.expander(f"❌ Decline {row['loan_id']}"):
                        reason = st.selectbox(
                            f"Select reason for rejecting {row['loan_id']}",
                            ["Low credit score", "Incomplete documentation", "Not proper certification", "Unstable income", "Other"],
                            key=f"reason_select_{row['loan_id']}"
                        )
                        if reason == "Other":
                            reason_custom = st.text_input("Enter custom reason", key=f"reason_custom_{row['loan_id']}")
                            final_reason = reason_custom.strip() if reason_custom.strip() else "Admin-declined"
                        else:
                            final_reason = reason

                        if st.button(f"Confirm Rejection for {row['loan_id']}", key=f"confirm_reject_{row['loan_id']}"):
                            loans_df.loc[loans_df["loan_id"] == row["loan_id"], "status"] = "declined"
                            loans_df.loc[loans_df["loan_id"] == row["loan_id"], "remarks"] = f"{final_reason}. Risk Score: {risk_score}%"
                            loan_status_df.loc[loan_status_df["loan_id"] == row["loan_id"], "status"] = "declined"
                            loan_status_df.loc[loan_status_df["loan_id"] == row["loan_id"], "remarks"] = f"{final_reason}. Risk Score: {risk_score}%"
                            save_csv(loans_df, loans_file)
                            save_csv(loan_status_df, loan_status_file)
                            st.session_state.loan_action_taken = True

            if st.session_state.get("loan_action_taken", False):
                st.session_state.loan_action_taken = False
                st.rerun()


    elif option == "🔍 Fetch User Info":
        st.subheader("Fetch User Details")
        username_input = st.text_input("Enter Username")
        if st.button("Fetch Info"):
            user_info = users_df[users_df["username"] == username_input]
            if user_info.empty:
                st.error("User not found.")
            else:
                user_id = user_info.iloc[0]["user_id"]
                account_info = accounts_df[accounts_df["user_id"] == user_id]
                transaction_info = transactions_df[transactions_df["user_id"] == user_id]
                loan_info = loans_df[loans_df["user_id"] == user_id]

                st.write("👤 User Info", user_info.drop(columns=['password'], errors='ignore'))
                st.write("🏦 Account Info", account_info)
                st.write("💸 Transaction History", transaction_info)
                st.write("📄 Loan History", loan_info)

                # 📎 Show uploaded documents
                st.write("📎 Uploaded Loan Documents")
                for _, loan in loan_info.iterrows():
                    loan_id = loan["loan_id"]
                    st.markdown(f"#### Loan ID: `{loan_id}`")
                    doc_path = os.path.join("documents", loan_id)

                    if os.path.isdir(doc_path):
                        files = os.listdir(doc_path)
                        if files:
                            for file in files:
                                file_path = os.path.join(doc_path, file)
                                with open(file_path, "rb") as f:
                                    file_bytes = f.read()
                                    st.download_button(
                                        label=f"📄 {file}",
                                        data=file_bytes,
                                        file_name=file,
                                        mime="application/octet-stream",
                                        key=f"download_{loan_id}_{file}"
                                    )

                        else:
                            st.info("No documents uploaded for this loan.")
                    else:
                        st.warning("📁 Document folder not found for this loan.")

    elif option == "📊 Loan Summary & Analytics":
        st.subheader("📊 Loan Analytics Dashboard")
        loans_df["application_date"] = pd.to_datetime(loans_df["application_date"], errors='coerce')
        start_date, end_date = st.date_input("Select Date Range", [loans_df["application_date"].min(), loans_df["application_date"].max()])
        filtered = loans_df[(loans_df["application_date"] >= pd.to_datetime(start_date)) &
                            (loans_df["application_date"] <= pd.to_datetime(end_date))]

        if filtered.empty:
            st.info("No loan applications found in this date range.")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Loans", len(filtered))
        col2.metric("Approved", (filtered["status"] == "approved").sum())
        col3.metric("Declined", (filtered["status"] == "declined").sum())

        csv = filtered.to_csv(index=False)
        st.download_button("📥 Download Filtered Loan Data", csv, "loan_summary.csv", "text/csv")

        monthly = filtered.groupby([filtered["application_date"].dt.to_period("M"), "status"]).size().unstack().fillna(0)
        monthly.index = monthly.index.astype(str)
        st.write("### 📈 Monthly Loan Approval Trends")
        fig1, ax1 = plt.subplots()
        monthly.plot(ax=ax1, marker='o')
        ax1.set_title("Loan Status Over Time")
        st.pyplot(fig1)

        st.write("### ✅ Low Risk People (Auto-Approved Loans with Low Risk Score)")
        low_risk_loans = filtered[
            (filtered["status"] == "approved") &
            (filtered["remarks"].str.contains("Auto-approved", na=False))
        ]
        if low_risk_loans.empty:
            st.info("No auto-approved low risk loans found.")
        else:
            display_cols = ["loan_id", "user_id", "amount", "income", "purpose", "application_date", "remarks"]
            st.dataframe(low_risk_loans[display_cols].sort_values("application_date", ascending=False).reset_index(drop=True))

        st.write("### 🎯 Loan Status by Purpose")
        purpose_summary = filtered.groupby(["purpose", "status"]).size().unstack().fillna(0)
        fig2, ax2 = plt.subplots()
        purpose_summary.plot(kind="bar", stacked=True, ax=ax2)
        ax2.set_title("Loan Purpose vs Status")
        st.pyplot(fig2)


# User Dashboard
def user_dashboard():
    global loans_df, loan_status_df
    import google.generativeai as genai
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-pro')
   
    st.sidebar.title("User Menu")
    user_id = st.session_state.user["user_id"]
    accounts_df = st.session_state.accounts_df
    loans_df = st.session_state.loans_df
    transactions_df = st.session_state.transactions_df

    # Dark/Light Mode Toggle
    
    choice = st.sidebar.radio("Go to", [
        "📈 Account Summary",
        "📝 Apply for Loan",
        "📊 Loan Status",
        "💵 Transactions",
        "🏦 Transfer Between Accounts",
        "💳 Pay Monthly EMI",
        "📚 Loan Repayment History",
        "🤖 AI Assistant Help"
    ])
    
    if choice == "📈 Account Summary":
        st.subheader("Account Summary")
    
        acc = accounts_df[accounts_df["user_id"] == user_id]
        if not acc.empty:
            st.dataframe(acc)
        else:
            st.info("No account information found.")
    
        user_loans = loans_df[loans_df["user_id"] == user_id]
        approved_loans = user_loans[user_loans["status"] == "approved"]
        num_loans = len(approved_loans)
        total_income = user_loans["income"].max() if not user_loans.empty else 0
    
        repayments = transactions_df[(transactions_df["user_id"] == user_id) & (transactions_df["loan_id"] != "")]
        total_repaid = repayments["amount"].sum()
    
        user_transactions = transactions_df[transactions_df["user_id"] == user_id]
        num_transactions = len(user_transactions)
    
        score = 50
        if total_income > 50000:
            score += 15
        elif total_income > 20000:
            score += 10
        elif total_income > 10000:
            score += 5
    
        if num_loans >= 3:
            score += 10
        elif num_loans == 2:
            score += 5
    
        if total_repaid > 0:
            score += min(15, total_repaid / 10000 * 5)
    
        if num_transactions > 10:
            score += 10
        elif num_transactions > 5:
            score += 5
    
        score = min(100, score)
        grade = "Excellent" if score >= 80 else "Good" if score >= 60 else "Fair" if score >= 40 else "Poor"
    
        if "credit_history" not in st.session_state:
            st.session_state.credit_history = []
        st.session_state.credit_history.append(score)
    
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Monthly Income", f"₹{total_income}")
        col2.metric("✅ Loans Approved", num_loans)
        col3.metric("💳 Total Repaid", f"₹{total_repaid}")
        col4.metric("🔄 Transactions", num_transactions)
    
        st.markdown(f"""
            <div style='text-align: center; margin-top: 20px;'>
                <h2 style='color: #4CAF50;'>📊 Your Credit Score: {score} / 100 ({grade})</h2>
                <progress value='{score}' max='100' style='width: 80%; height: 25px;'></progress>
            </div>
        """, unsafe_allow_html=True)
    
        if len(st.session_state.credit_history) > 1:
            st.line_chart(st.session_state.credit_history)
    
        if not approved_loans.empty:
            loan = approved_loans.iloc[0]
            app_date = pd.to_datetime(loan["application_date"], errors='coerce')
            paid = transactions_df[(transactions_df["user_id"] == user_id) & (transactions_df["loan_id"] == loan["loan_id"])].shape[0]
            next_due = app_date + pd.DateOffset(months=paid)
            st.info(f"📅 Next EMI Due: {next_due.strftime('%Y-%m-%d')} (Loan ID: {loan['loan_id']})")
        else:
            st.info("No active loans. Apply now to build your credit!")
    
        if not user_transactions.empty:
            user_transactions["date"] = pd.to_datetime(user_transactions["date"], errors='coerce')
            tx_monthly = user_transactions.groupby(user_transactions["date"].dt.to_period("M"))["amount"].sum()
            tx_monthly.index = tx_monthly.index.astype(str)
            st.write("### 📊 Monthly Transaction Summary")
            st.bar_chart(tx_monthly)
        else:
            st.write("No transactions to display.")

    elif choice == "📝 Apply for Loan":
        st.subheader("Loan Application Form")

        aadhaar_input = st.text_input("Enter your Aadhaar Number to verify")
        verified = False


        if st.button("Verify Aadhaar"):
            user_aadhaar = accounts_df[accounts_df["user_id"] == user_id]["aadhar"].values[0] if "aadhar" in accounts_df.columns else None
            if user_aadhaar and str(user_aadhaar) == aadhaar_input:
                st.success("✅ Aadhaar verified successfully.")
                st.session_state.aadhaar_verified = True
            else:
                st.error("❌ Aadhaar verification failed.")
                st.session_state.aadhaar_verified = False
    
        if st.session_state.get("aadhaar_verified", False):
            st.markdown("### 📎 Upload Required Documents")
            id_proof = st.file_uploader("Identity Proof (PAN, Voter ID, etc.)", type=["pdf", "jpg", "png"])
            address_proof = st.file_uploader("Address Proof", type=["pdf", "jpg", "png"])
            income_proof = st.file_uploader("Income Proof", type=["pdf", "jpg", "png"])
            bank_statement = st.file_uploader("Bank Statement", type=["pdf", "jpg", "png"])
    
            amount = st.number_input("Loan Amount", min_value=1000)
            purpose = st.selectbox("Purpose", ["Education", "Medical", "Home Renovation", "Vehicle", "Business", "Personal"])
            income = st.number_input("Monthly Income", min_value=0)
    
            all_docs_uploaded = id_proof and address_proof and income_proof and bank_statement
    
            if st.button("Submit Application"):
                if not all_docs_uploaded:
                    st.warning("⚠️ Please upload all required documents before submission.")
                else:
                    train_df = loans_df[loans_df["status"] != "pending"]
                    if train_df.shape[0] < 10 or len(train_df["status"].unique()) < 2:
                        decision = "pending"
                        remarks = "Awaiting admin review"
                    else:
                        train_df = train_df[["amount", "income", "status"]].dropna()
                        X_train = train_df[["amount", "income"]]
                        y_train = (train_df["status"] == "approved").astype(int)
    
                        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                        model.fit(X_train, y_train)
    
                        X_new = np.array([[amount, income]])
                        prob = model.predict_proba(X_new)[0][1]
                        risk_score = round((1 - prob) * 100, 2)
    
                        if risk_score <= 39:
                            decision = "approved"
                            remarks = f"Auto-approved. Risk Score: {risk_score}%"
                        elif risk_score >= 61:
                            auto_reason = random.choice([
                                "Low credit score based on prior history",
                                "Insufficient income compared to requested amount",
                                "Debt-to-income ratio too high",
                                "Missing financial documentation",
                                "Loan amount exceeds eligibility"
                            ])
                            decision = "declined"
                            remarks = f"Auto-declined: {auto_reason}. Risk Score: {risk_score}%"
                        else:
                            decision = "pending"
                            remarks = f"Awaiting admin review. Risk Score: {risk_score}%"
    
                    loan_id = f"L{len(loans_df)+1:03d}"
                    new_loan = {
                        "loan_id": loan_id,
                        "user_id": user_id,
                        "amount": amount,
                        "purpose": purpose,
                        "income": income,
                        "status": decision,
                        "application_date": pd.Timestamp.today().strftime('%Y-%m-%d'),
                        "remarks": remarks
                    }
    
                    loans_df.loc[len(loans_df)] = new_loan
                    loan_status_df.loc[len(loan_status_df)] = new_loan
                    save_csv(loans_df, loans_file)
                    save_csv(loan_status_df, loan_status_file)
    
                    doc_folder = os.path.join("documents", loan_id)
                    os.makedirs(doc_folder, exist_ok=True)
    
                    def save_file(uploaded_file, name):
                        with open(os.path.join(doc_folder, name), "wb") as f:
                            f.write(uploaded_file.read())
    
                    save_file(id_proof, "identity_" + id_proof.name)
                    save_file(address_proof, "address_" + address_proof.name)
                    save_file(income_proof, "income_" + income_proof.name)
                    save_file(bank_statement, "bank_" + bank_statement.name)
    
                    # ✅ Email Notification
                    user_row = users_df[users_df["user_id"] == user_id]
                    user_email = user_row["email"].values[0] if "email" in user_row.columns else None
                    user_name = user_row["username"].values[0]
                    if user_email:
                        send_loan_email(user_email, user_name, loan_id, decision, remarks)
                    else:
                        st.warning("User email not found. Email notification skipped.")
    
                    st.success(f"Loan Application Submitted! Status: **{decision.capitalize()}**")
                    st.info(f"📝 Remarks: {remarks}")



    elif choice == "📊 Loan Status":
        st.subheader("Your Loan Applications")
        user_loans = loans_df[loans_df["user_id"] == user_id]
        st.dataframe(user_loans)

    elif choice == "💵 Transactions":
        st.subheader("Transactions")
        tx = transactions_df[transactions_df["user_id"] == user_id]
        st.dataframe(tx)

    elif choice == "🏦 Transfer Between Accounts":
        st.subheader("Transfer Amount to Another Account")
        sender_account = accounts_df[accounts_df["user_id"] == user_id].iloc[0]
        sender_balance = sender_account["balance"]
        sender_account_no = sender_account["account_number"]

        st.write(f"💳 Your Account Number: `{sender_account_no}`")
        st.write(f"💰 Your Current Balance: ₹{sender_balance}")

        recipient_account_no = st.text_input("Recipient Account Number")

        if recipient_account_no:
            recipient_row = accounts_df[accounts_df["account_number"] == recipient_account_no]
            if not recipient_row.empty:
                recipient_user_id = recipient_row.iloc[0]["user_id"]
                recipient_user = users_df[users_df["user_id"] == recipient_user_id].iloc[0]
                recipient_name = recipient_user["username"]
                recipient_mobile = recipient_row.iloc[0]["mobile"]
                st.info(f"👤 Recipient Name: **{recipient_name}**\n📱 Mobile: **{recipient_mobile}**")
            else:
                st.warning("⚠️ No user found with this account number.")

        transfer_amount = st.number_input("Amount to Transfer", min_value=1.0)
        payment_method = st.radio("Select Payment Method", ["UPI", "Net Banking", "Bank Transfer"])
        entered_password = st.text_input("Enter your password to confirm", type="password")

        if st.button("Transfer"):
            actual_password = users_df[users_df["user_id"] == user_id].iloc[0]["password"]
            if not recipient_account_no:
                st.warning("Please enter a valid recipient account number.")
            elif recipient_account_no == sender_account_no:
                st.error("❌ You cannot transfer to your own account.")
            elif recipient_account_no not in accounts_df["account_number"].values:
                st.error("❌ Recipient account not found.")
            elif transfer_amount > sender_balance:
                st.error("❌ Insufficient balance.")
            elif entered_password != actual_password:
                st.error("❌ Incorrect password.")
            else:
                accounts_df.loc[accounts_df["user_id"] == user_id, "balance"] -= transfer_amount
                accounts_df.loc[accounts_df["account_number"] == recipient_account_no, "balance"] += transfer_amount
                save_csv(accounts_df, accounts_file)

                sender_tx = {
                    "user_id": user_id,
                    "loan_id": "",
                    "amount": -transfer_amount,
                    "method": f"Transfer Out ({payment_method})",
                    "date": pd.Timestamp.today().strftime('%Y-%m-%d')
                }
                recipient_user_id = accounts_df[accounts_df["account_number"] == recipient_account_no].iloc[0]["user_id"]
                recipient_tx = {
                    "user_id": recipient_user_id,
                    "loan_id": "",
                    "amount": transfer_amount,
                    "method": f"Transfer In ({payment_method})",
                    "date": pd.Timestamp.today().strftime('%Y-%m-%d')
                }
                transactions_df.loc[len(transactions_df)] = sender_tx
                transactions_df.loc[len(transactions_df)] = recipient_tx
                save_csv(transactions_df, transactions_file)

                new_balance = accounts_df[accounts_df["user_id"] == user_id].iloc[0]["balance"]
                st.success(f"✅ ₹{transfer_amount} transferred to `{recipient_account_no}` successfully!")
                st.info(f"💰 Updated Balance: ₹{new_balance}")

    elif choice == "💳 Pay Monthly EMI":
        st.subheader("Pay Monthly EMI")
        approved_loans = loans_df[(loans_df["user_id"] == user_id) & (loans_df["status"] == "approved")]
        if approved_loans.empty:
            st.info("No approved loans found.")
            return

        selected_loan_id = st.selectbox("Select Loan ID", approved_loans["loan_id"])
        loan = approved_loans[approved_loans["loan_id"] == selected_loan_id].iloc[0]
        loan_amount = loan["amount"]
        application_date = pd.to_datetime(loan["application_date"])
        tenure = 12
        interest = 10 / 100 / 12
        emi = round((loan_amount * interest * (1 + interest) ** tenure) / ((1 + interest) ** tenure - 1), 2)

        paid_emis = transactions_df[(transactions_df["user_id"] == user_id) & (transactions_df["loan_id"] == selected_loan_id)]
        paid_count = paid_emis.shape[0]
        remaining = tenure - paid_count

        st.write(f"📄 Loan: ₹{loan_amount}  \n💰 EMI: ₹{emi}  \n📆 Remaining: {remaining} of {tenure}")

        if remaining == 0:
            st.success("🎉 Loan already paid in full.")
            return

        method = st.radio("Payment Method", ["UPI", "Net Banking"])
        if st.button("Pay EMI"):
            new_tx = {
                "user_id": user_id,
                "loan_id": selected_loan_id,
                "amount": emi,
                "method": method,
                "date": pd.Timestamp.today().strftime('%Y-%m-%d')
            }
            transactions_df.loc[len(transactions_df)] = new_tx
            save_csv(transactions_df, transactions_file)
            st.success(f"✅ EMI of ₹{emi} paid successfully.")
            st.rerun()

        st.write("### 🗓️ EMI Payment Schedule")
        schedule = []
        for i in range(tenure):
            due = application_date + pd.DateOffset(months=i)
            status = "Paid" if i < paid_count else ("Due" if i == paid_count else "Upcoming")
            schedule.append({
                "Installment #": i + 1,
                "Due Date": due.date(),
                "EMI Amount": emi,
                "Status": status
            })
        st.dataframe(pd.DataFrame(schedule))

    elif choice == "📚 Loan Repayment History":
        st.subheader("Loan Repayment History")
        tx = transactions_df[transactions_df["user_id"] == user_id]
        if tx.empty:
            st.info("No repayments made yet.")
        else:
            st.dataframe(tx)
            summary = tx.groupby("loan_id")["amount"].sum().reset_index().rename(columns={"amount": "Total Paid"})
            st.write("### Summary by Loan")
            st.dataframe(summary)

    elif choice == "🤖 AI Assistant Help":

# --- AI Assistant Help Section ---
         st.subheader("🤖 AI Chat Assistant")
         st.markdown("Ask any questions related to your account, EMI, transfers, etc.")

# Configure Gemini API
         try:
         genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
         except KeyError:
            st.error("🚨 GEMINI_API_KEY not found in Streamlit secrets. Please add it to your `secrets.toml` file.")
            st.stop() # Stop execution if API key is missing

# Initialize the Generative Model
         try:
            model = genai.GenerativeModel("gemini-pro")
         except Exception as e:
            st.error(f"Failed to initialize Gemini model: {e}")
            st.stop()

# Initialize chat history in session state
         if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

# Display previous chat messages
         for user_input, bot_reply in st.session_state.chat_history:
            st.markdown(f"**🧑 You:** {user_input}")
            st.markdown(f"**🤖 Assistant:** {bot_reply}")

# User input text area
         question = st.text_input("Type your question here...")

# Ask button
         if st.button("Ask"):
            if question.strip():
               try:
            # Generate content using the Gemini Pro model
                response = model.generate_content(question)
                reply = response.text.strip()
               except Exception as e:
                reply = f"⚠️ Oops! Something went wrong: {e}. Please try again."

        # Append to chat history
               st.session_state.chat_history.append((question, reply))
               st.rerun() # Rerun to display the new message
            else:
               st.warning("Please enter a question to get assistance.")

# --- End of AI Assistant Help Section ---

    


# Main App Logic
if st.session_state.user:
    st.sidebar.write(f"👋 Welcome, {st.session_state.user['username']}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()
    if st.session_state.user.get("role") == "admin":
        admin_dashboard()
    else:
        user_dashboard()
else:
    login()
