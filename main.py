def user_dashboard():
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    st.sidebar.title("User Menu")
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
    user_id = st.session_state.user["user_id"]

    if choice == "📈 Account Summary":
        st.subheader("Account Summary")
        acc = accounts_df[accounts_df["user_id"] == user_id]
        st.dataframe(acc)

    elif choice == "📝 Apply for Loan":
        st.subheader("Loan Application Form")
        amount = st.number_input("Loan Amount", min_value=1000)
        purpose = st.selectbox("Purpose", ["Education", "Medical", "Home Renovation", "Vehicle", "Business", "Personal"])
        income = st.number_input("Monthly Income", min_value=0)
        if st.button("Submit Application"):
            loan_id = f"L{len(loans_df)+1:03d}"
            new_loan = {
                "loan_id": loan_id,
                "user_id": user_id,
                "amount": amount,
                "purpose": purpose,
                "income": income,
                "status": "pending",
                "application_date": pd.Timestamp.today().strftime('%Y-%m-%d'),
                "remarks": "Awaiting review"
            }
            loans_df_updated = pd.concat([loans_df, pd.DataFrame([new_loan])], ignore_index=True)
            save_csv(loans_df_updated, loans_file)
            save_csv(loans_df_updated, loan_status_file)
            st.success("Loan Application Submitted!")

    elif choice == "📊 Loan Status":
        st.subheader("Your Loan Applications")
        user_loans = loans_df[loans_df["user_id"] == user_id]
        st.dataframe(user_loans)

    elif choice == "💵 Transactions":
        st.subheader("Transaction History")
        tx = transactions_df[transactions_df["user_id"] == user_id]
        st.dataframe(tx)

    elif choice == "🏦 Transfer Between Accounts":
        st.subheader("Transfer Amount to Another Account")
        sender_account = accounts_df[accounts_df["user_id"] == user_id].iloc[0]
        sender_balance = sender_account["balance"]
        sender_account_no = sender_account["account_no"]

        st.write(f"💳 Your Account Number: `{sender_account_no}`")
        st.write(f"💰 Your Current Balance: ₹{sender_balance}")

        recipient_account_no = st.text_input("Recipient Account Number")

        if recipient_account_no:
            recipient_row = accounts_df[accounts_df["account_no"] == recipient_account_no]
            if not recipient_row.empty:
                recipient_user_id = recipient_row.iloc[0]["user_id"]
                recipient_user = users_df[users_df["user_id"] == recipient_user_id].iloc[0]
                st.info(f"👤 Recipient Name: **{recipient_user['username']}**  \n📱 Mobile: **{recipient_row.iloc[0]['mobile']}**")
            else:
                st.warning("⚠️ No user found with this account number.")

        transfer_amount = st.number_input("Amount to Transfer", min_value=1.0)
        payment_method = st.radio("Select Payment Method", ["UPI", "Net Banking", "Bank Transfer"])
        entered_password = st.text_input("Enter your password to confirm", type="password")

        if st.button("Transfer"):
            actual_password = users_df[users_df["user_id"] == user_id].iloc[0]["password"]
            if not recipient_account_no:
                st.warning("Please enter a recipient account number.")
            elif recipient_account_no == sender_account_no:
                st.error("❌ Cannot transfer to your own account.")
            elif recipient_account_no not in accounts_df["account_no"].values:
                st.error("❌ Invalid recipient account.")
            elif transfer_amount > sender_balance:
                st.error("❌ Insufficient balance.")
            elif entered_password != actual_password:
                st.error("❌ Incorrect password.")
            else:
                accounts_df.loc[accounts_df["user_id"] == user_id, "balance"] -= transfer_amount
                accounts_df.loc[accounts_df["account_no"] == recipient_account_no, "balance"] += transfer_amount
                save_csv(accounts_df, accounts_file)

                sender_tx = {
                    "user_id": user_id,
                    "loan_id": "",
                    "amount": -transfer_amount,
                    "method": f"Transfer Out ({payment_method})",
                    "date": pd.Timestamp.today().strftime('%Y-%m-%d')
                }
                recipient_user_id = accounts_df[accounts_df["account_no"] == recipient_account_no].iloc[0]["user_id"]
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
                st.info(f"💰 New Balance: ₹{new_balance}")

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
        st.subheader("🤖 AI Chat Assistant")
        st.markdown("Ask any questions related to your account, EMI, transfers, etc.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for user_input, bot_reply in st.session_state.chat_history:
            st.markdown(f"**🧑 You:** {user_input}")
            st.markdown(f"**🤖 Assistant:** {bot_reply}")

        question = st.text_input("Type your question here...")
        if st.button("Ask"):
            if question.strip():
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful banking assistant."},
                            {"role": "user", "content": question}
                        ]
                    )
                    reply = response.choices[0].message.content.strip()
                except Exception as e:
                    reply = f"⚠️ Error: {e}"

                st.session_state.chat_history.append((question, reply))
                st.rerun()
            else:
                st.warning("Please enter a question.")
