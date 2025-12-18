# lite_12.py
# If you get DB schema errors once, delete expenses.db and restart.

import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
from datetime import datetime, date
import bcrypt
import ollama

DB_FILE = "expenses.db"

# ---------------- DB INIT ----------------
def initialize_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Users
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        );
        """
    )

    # Expenses
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            payment_mode TEXT,
            description TEXT,
            amount_paid REAL NOT NULL,
            cashback REAL,
            FOREIGN KEY (username) REFERENCES users (username)
        );
        """
    )

    # Budgets
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            month TEXT NOT NULL,     -- 'YYYY-MM'
            amount REAL NOT NULL,
            UNIQUE (username, month)
        );
        """
    )

    # Gamification
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS gamification (
            username TEXT PRIMARY KEY,
            last_entry_date TEXT,
            streak_days INTEGER DEFAULT 0,
            lifetime_saved REAL DEFAULT 0.0
        );
        """
    )

    conn.commit()
    conn.close()

initialize_db()

# ---------------- DB HELPERS ----------------
@st.cache_data(ttl=60)
def fetch_expenses(username: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        "SELECT * FROM expenses WHERE username = ? ORDER BY date DESC",
        conn,
        params=(username,),
    )
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

def insert_expense(username, date_str, category, payment_mode, description, amount_paid, cashback):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO expenses (username, date, category, payment_mode, description, amount_paid, cashback)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (username, date_str, category, payment_mode, description, amount_paid, cashback),
    )
    conn.commit()
    conn.close()
    fetch_expenses.clear()

def delete_expense(expense_id: int):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("DELETE FROM expenses WHERE id = ?;", (expense_id,))
    conn.commit()
    conn.close()
    fetch_expenses.clear()

def set_budget(username: str, month: str, amount: float):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO budgets (username, month, amount)
        VALUES (?, ?, ?)
        ON CONFLICT(username, month) DO UPDATE SET amount = excluded.amount;
        """,
        (username, month, amount),
    )
    conn.commit()
    conn.close()

def get_budget(username: str, month: str):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "SELECT amount FROM budgets WHERE username = ? AND month = ?;",
        (username, month),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def list_budgets(username: str):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        "SELECT month, amount FROM budgets WHERE username = ? ORDER BY month;",
        conn,
        params=(username,),
    )
    conn.close()
    return df

def delete_budget(username: str, month: str):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM budgets WHERE username = ? AND month = ?;",
        (username, month),
    )
    conn.commit()
    conn.close()

# ---- Gamification helpers ----
def get_gamification_row(username: str):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "SELECT last_entry_date, streak_days, lifetime_saved FROM gamification WHERE username = ?;",
        (username,),
    )
    row = cur.fetchone()
    conn.close()
    return row

def update_gamification(username: str, today_str: str, saved_delta: float):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    row = get_gamification_row(username)
    if row is None:
        streak = 1
        lifetime_saved = max(saved_delta, 0.0)
        cur.execute(
            """
            INSERT INTO gamification (username, last_entry_date, streak_days, lifetime_saved)
            VALUES (?, ?, ?, ?);
            """,
            (username, today_str, streak, lifetime_saved),
        )
    else:
        last_entry_str, streak, lifetime_saved = row
        last_date = datetime.strptime(last_entry_str, "%Y-%m-%d").date() if last_entry_str else None
        today_date = datetime.strptime(today_str, "%Y-%m-%d").date()
        if last_date is None:
            streak = 1
        else:
            delta_days = (today_date - last_date).days
            if delta_days == 0:
                pass
            elif delta_days == 1:
                streak += 1
            else:
                streak = 1
        lifetime_saved = float(lifetime_saved) + max(saved_delta, 0.0)
        cur.execute(
            """
            UPDATE gamification
            SET last_entry_date = ?, streak_days = ?, lifetime_saved = ?
            WHERE username = ?;
            """,
            (today_str, streak, lifetime_saved, username),
        )
    conn.commit()
    conn.close()

# ---------------- PASSWORD HASHING & AUTH ----------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

def create_user(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password) VALUES (?, ?);",
            (username, hash_password(password)),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT password FROM users WHERE username = ?;", (username,))
    row = cur.fetchone()
    conn.close()
    return bool(row and check_password(password, row[0]))

# ---------------- SESSION STATE ----------------
st.set_page_config(
    page_title="💰 Live Expense Tracker Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "page" not in st.session_state:
    st.session_state.page = "login"
if "ollama_chat_open" not in st.session_state:
    st.session_state.ollama_chat_open = False
if "ollama_messages" not in st.session_state:
    st.session_state.ollama_messages = []

def ask_ollama(messages, model_name: str = "llama3"):
    prompt_parts = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    prompt_parts.append("Assistant:")
    full_prompt = "\n".join(prompt_parts)
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": full_prompt}],
    )
    return response["message"]["content"].strip()

# ---------------- AUTH UI ----------------
def login_page():
    st.title("🔑 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if verify_user(u, p):
            st.session_state.logged_in = True
            st.session_state.username = u
            st.session_state.page = "main"
            st.rerun()
        else:
            st.error("Invalid username or password.")

def signup_page():
    st.title("📝 Sign Up")
    u = st.text_input("Choose a Username")
    p = st.text_input("Choose a Password", type="password")
    if st.button("Sign Up"):
        if create_user(u, p):
            st.success("Account created successfully! Please log in.")
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error("Username already exists. Please choose another.")

if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        signup_page()
    st.sidebar.button("Go to Sign Up", on_click=lambda: setattr(st.session_state, "page", "signup"))
    st.sidebar.button("Go to Login", on_click=lambda: setattr(st.session_state, "page", "login"))
    st.stop()

# ---------------- MAIN APP LAYOUT ----------------
st.sidebar.button("Logout", on_click=lambda: setattr(st.session_state, "logged_in", False))
st.sidebar.info(f"👤 Logged in as: **{st.session_state.username}**")

st.title("💰 Live Expense Tracker Dashboard")
st.write("Track and visualize your spending with global XP, levels, streaks, monthly quests and custom graphs.")
st.markdown("---")
st.sidebar.markdown("---")

if st.sidebar.button("🤖 Open Local AI Chat (Ollama)"):
    st.session_state.ollama_chat_open = True

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("➕ Enter New Expense")

with st.sidebar.form("new_expense_form", clear_on_submit=True):
    date_input = st.date_input("🗓 Date", date.today())
    category_options = [
        "Food", "Transport", "Bills", "Entertainment", "Shopping",
        "Healthcare", "Travel", "Subscriptions", "Other",
    ]
    category_input = st.selectbox("🏷 Category", category_options)
    payment_modes = ["Cash", "Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"]
    payment_mode_input = st.selectbox("💳 Payment Mode", payment_modes)
    amount_input = st.number_input("💵 Amount Paid", min_value=1.0, step=10.0, format="%.2f")
    description_input = st.text_input("📝 Description/Note (e.g., 'Dinner at Italian Place')")
    cashback_input = st.number_input("🎁 Cashback/Reward", min_value=0.0, step=1.0, format="%.2f", value=0.0)
    submitted = st.form_submit_button("Submit Expense")

if submitted:
    insert_expense(
        st.session_state.username,
        str(date_input),
        category_input,
        payment_mode_input,
        description_input,
        amount_input,
        cashback_input,
    )
    st.success("✅ Expense added successfully!")

# --------- BUDGET MANAGER ----------
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Budget Manager")

budget_df = list_budgets(st.session_state.username)
exp_df_for_months = fetch_expenses(st.session_state.username)
expense_months = []
if not exp_df_for_months.empty:
    tmp = exp_df_for_months.copy()
    tmp["YearMonth"] = tmp["date"].apply(lambda x: x.strftime("%Y-%m"))
    expense_months = sorted(tmp["YearMonth"].unique())

all_months_set = set(budget_df["month"].tolist()) | set(expense_months)
if not all_months_set:
    all_months_set.add(datetime.today().strftime("%Y-%m"))
all_months = sorted(all_months_set)

selected_budget_month = st.sidebar.selectbox("Select month to edit", all_months)
current_budget_val = get_budget(st.session_state.username, selected_budget_month) or 0.0

new_budget_val = st.sidebar.number_input(
    f"Budget for {selected_budget_month}",
    min_value=0.0,
    step=1000.0,
    value=float(current_budget_val),
    format="%.2f",
)

col_b1, col_b2 = st.sidebar.columns(2)
if col_b1.button("Save / Update"):
    set_budget(st.session_state.username, selected_budget_month, float(new_budget_val))
    st.sidebar.success(f"Budget for {selected_budget_month} saved/updated!")
    st.rerun()

if col_b2.button("Delete Budget"):
    delete_budget(st.session_state.username, selected_budget_month)
    st.sidebar.warning(f"Budget for {selected_budget_month} deleted!")
    st.rerun()

st.sidebar.markdown("### Your Budgets")
budget_df = list_budgets(st.session_state.username)
if not budget_df.empty:
    st.sidebar.dataframe(budget_df, use_container_width=True)
else:
    st.sidebar.caption("No budgets saved yet.")

# ---------------- LOAD DATA ----------------
df = fetch_expenses(st.session_state.username)
if df.empty:
    st.info("No expenses recorded yet. Use the sidebar to add your first transaction!")
    st.stop()

# ---------------- METRICS ----------------
total_spent = df["amount_paid"].sum()
total_cashback = df["cashback"].sum()
num_transactions = len(df)

st.subheader("Key Financial Metrics")
c1, c2, c3 = st.columns(3)
c1.metric("Total Spent", f"₹ {total_spent:,.2f}")
c2.metric("Total Cashback Earned", f"₹ {total_cashback:,.2f}")
c3.metric("Total Transactions", f"{num_transactions}")

st.markdown("---")

# ---------------- GLOBAL XP / BADGES ----------------
df["YearMonth"] = df["date"].apply(lambda x: x.strftime("%Y-%m"))
budgets_all = list_budgets(st.session_state.username)

saved_lifetime_delta = 0.0
if not budgets_all.empty:
    for _, row in budgets_all.iterrows():
        ym = row["month"]
        b_amt = row["amount"]
        spent_m = df.loc[df["YearMonth"] == ym, "amount_paid"].sum()
        saved_lifetime_delta += (b_amt - spent_m)

total_budget_all = budgets_all["amount"].sum() if not budgets_all.empty else 0.0
total_spent_all = df["amount_paid"].sum()

if total_budget_all > 0:
    saved_ratio = saved_lifetime_delta / total_budget_all
else:
    saved_ratio = 0.0

if saved_ratio >= 0:
    base_xp = int(min(saved_ratio, 1.0) * 5000)
else:
    base_xp = max(0, int(3000 + saved_ratio * 5000))

badge = "🥉 Starter"
rank_text = "New Saver"
if saved_ratio >= 0.3:
    badge = "🏅 Gold Saver"
    rank_text = "Elite Saver"
elif saved_ratio >= 0.1:
    badge = "🥈 Silver Saver"
    rank_text = "Disciplined Planner"
elif saved_ratio >= 0:
    badge = "🥉 Bronze Saver"
    rank_text = "Balanced Spender"
else:
    badge = "⚠️ Overspender"
    rank_text = "Needs Improvement"

current_month_for_alert = selected_budget_month
cur_month_spent = df.loc[df["YearMonth"] == current_month_for_alert, "amount_paid"].sum()
budget_amount_current = get_budget(st.session_state.username, current_month_for_alert)

if budget_amount_current and budget_amount_current > 0:
    today_str = datetime.today().strftime("%Y-%m-%d")
    if cur_month_spent > budget_amount_current:
        st.error("🚨 You overspent this month!")
        st.markdown(
            f"**Details:** Spent **₹{cur_month_spent:,.2f}** out of a budget of **₹{budget_amount_current:,.2f}**."
        )
    elif cur_month_spent > 0.9 * budget_amount_current:
        st.warning("⚠️ You are close to your monthly budget.")
        st.markdown(
            f"**Details:** Spent **₹{cur_month_spent:,.2f}** out of **₹{budget_amount_current:,.2f}**."
        )
    else:
        st.success("✅ Great! You are within your monthly budget.")
        st.markdown(
            f"**Details:** Spent **₹{cur_month_spent:,.2f}** out of **₹{budget_amount_current:,.2f}**."
        )
    update_gamification(st.session_state.username, today_str, saved_delta=saved_lifetime_delta)
else:
    st.info("Set at least one monthly budget in the sidebar to enable alerts and better XP scaling.")

g_row = get_gamification_row(st.session_state.username)
streak_days = g_row[1] if g_row else 0
lifetime_saved_recorded = float(g_row[2]) if g_row else 0.0

# ---------------- MONTHLY QUESTS ----------------
st.subheader("🎯 Savings Quests (New each month)")
this_month_df = df[df["YearMonth"] == current_month_for_alert]
quest_msgs = []
quests_completed = 0
QUEST_XP_PER_QUEST = 300

if not this_month_df.empty:
    total_this_month = this_month_df["amount_paid"].sum()
    food_spent = this_month_df[this_month_df["category"] == "Food"]["amount_paid"].sum()
    food_ratio = food_spent / total_this_month if total_this_month > 0 else 0
    if food_ratio <= 0.3:
        quest_msgs.append("✅ Quest 1 (this month): Keep Food spending ≤ 30% of total.")
        quests_completed += 1
    else:
        quest_msgs.append("❌ Quest 1 (this month): Food spending > 30%. Try reducing it.")

    if len(this_month_df) >= 10:
        quest_msgs.append("✅ Quest 2 (this month): Log 10+ transactions.")
        quests_completed += 1
    else:
        quest_msgs.append("❌ Quest 2 (this month): Fewer than 10 transactions so far.")

    cb_this_month = this_month_df["cashback"].sum()
    if cb_this_month >= 500:
        quest_msgs.append("✅ Quest 3 (this month): Earn ₹500+ cashback.")
        quests_completed += 1
    else:
        quest_msgs.append("❌ Quest 3 (this month): Cashback < ₹500; use more reward offers.")
else:
    quest_msgs.append("Add some expenses this month to unlock this month's quests.")

for q in quest_msgs:
    st.write(q)

quest_xp = quests_completed * QUEST_XP_PER_QUEST
st.markdown(f"**Quests completed this month:** {quests_completed}/3 → **Quest XP:** {quest_xp}")

# ---------- Lifetime Quest History ----------
if st.button("Show Lifetime Quest History"):
    with st.expander("Lifetime Quest History and Savings", expanded=True):
        all_months_sorted = sorted(df["YearMonth"].unique())
        if not all_months_sorted:
            st.write("No months with expenses yet.")
        else:
            q1_cnt = q2_cnt = q3_cnt = 0
            total_quest_xp_all = 0.0
            total_budget_history = 0.0
            total_spent_history = 0.0
            rows = []

            for ym in all_months_sorted:
                m_df = df[df["YearMonth"] == ym]
                if m_df.empty:
                    continue
                total_m = m_df["amount_paid"].sum()
                food_m = m_df[m_df["category"] == "Food"]["amount_paid"].sum()
                food_ratio_m = food_m / total_m if total_m > 0 else 0
                cb_m = m_df["cashback"].sum()
                budget_m = get_budget(st.session_state.username, ym) or 0.0
                saved_m = budget_m - total_m

                total_budget_history += budget_m
                total_spent_history += total_m

                q1_done = food_ratio_m <= 0.3
                q2_done = len(m_df) >= 10
                q3_done = cb_m >= 500

                if q1_done:
                    q1_cnt += 1
                if q2_done:
                    q2_cnt += 1
                if q3_done:
                    q3_cnt += 1

                completed_this_month = sum([q1_done, q2_done, q3_done])
                total_quest_xp_all += completed_this_month * QUEST_XP_PER_QUEST

                rows.append(
                    {
                        "Month": ym,
                        "Budget (₹)": budget_m,
                        "Spent (₹)": total_m,
                        "Saved (₹)": saved_m,
                        "Quest1_Food≤30%": "✅" if q1_done else "❌",
                        "Quest2_10+Txns": "✅" if q2_done else "❌",
                        "Quest3_₹500+CB": "✅" if q3_done else "❌",
                        "Quest XP (month)": completed_this_month * QUEST_XP_PER_QUEST,
                    }
                )

            history_df = pd.DataFrame(rows)
            st.write("**Per‑month quest history:**")
            st.dataframe(history_df, use_container_width=True)
            st.markdown("---")
            st.write(f"**Total lifetime spending:** ₹{total_spent_history:,.2f}")
            st.write(f"**Total lifetime budget set:** ₹{total_budget_history:,.2f}")
            st.write(f"**Lifetime saved (budget − spent):** ₹{(total_budget_history - total_spent_history):,.2f}")
            st.markdown("---")
            st.write(f"**Quest 1 completed in:** {q1_cnt} month(s)")
            st.write(f"**Quest 2 completed in:** {q2_cnt} month(s)")
            st.write(f"**Quest 3 completed in:** {q3_cnt} month(s)")
            st.write(f"**Total quest XP earned all‑time:** {int(total_quest_xp_all)}")
            st.markdown("---")

# ---------- LEVELS BASED ON TOTAL XP ----------
total_xp = base_xp + quest_xp
level = total_xp // 500
xp_in_level = total_xp % 500
level_progress = min(xp_in_level / 500, 1.0)

milestone_badge = ""
if lifetime_saved_recorded >= 100000:
    milestone_badge = "👑 Legendary Saver (₹1L+ saved lifetime)"
elif lifetime_saved_recorded >= 50000:
    milestone_badge = "💎 Diamond Saver (₹50k+ saved lifetime)"
elif lifetime_saved_recorded >= 20000:
    milestone_badge = "🌟 Platinum Saver (₹20k+ saved lifetime)"

st.subheader("🏆 Savings XP, Level & Badges")
col_xp, col_rank, col_level = st.columns(3)

with col_xp:
    st.metric("Total XP (global + quests)", total_xp)
    st.write(f"Base XP from savings: **{base_xp}**")
    st.write(f"Quest XP this month: **{quest_xp}**")
    st.write(f"🔥 Daily Streak: **{streak_days} days**")

with col_rank:
    st.write("**Rank / Badge**")
    st.write(f"{rank_text} {badge}")
    if milestone_badge:
        st.write(milestone_badge)

with col_level:
    st.write(f"**Level:** {level}")
    st.progress(level_progress)
    st.caption("Progress to next level")

st.markdown("---")

# ---------- OLLAMA CHAT BOX ----------
if st.session_state.ollama_chat_open:
    st.subheader("🤖 Local AI Spending Coach (Ollama)")

    category_totals = (
        df.groupby("category")["amount_paid"]
        .sum()
        .sort_values(ascending=False)
    )
    top3 = category_totals.head(3)
    cat_summary_lines = [f"{cat}: ₹{val:,.2f}" for cat, val in top3.items()]
    top3_text = "; ".join(cat_summary_lines) if cat_summary_lines else "No category data available"

    for msg in st.session_state.ollama_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask the local AI about your spending, categories, budget, quests, etc.")
    if user_msg:
        st.session_state.ollama_messages.append({"role": "user", "content": user_msg})

        stats_context = (
            "System: You are a financial coach. "
            "When the user asks about spending categories or top categories, "
            "always answer using the exact category names and rupee amounts given here.\n"
            f"Lifetime total spent: ₹{total_spent_all:,.2f}.\n"
            f"Lifetime total budget set: ₹{total_budget_all:,.2f}.\n"
            f"Lifetime saved (budget − spent): ₹{saved_lifetime_delta:,.2f}.\n"
            f"Current level: {level}, current streak: {streak_days} days.\n"
            f"Top 3 spending categories (lifetime): {top3_text}.\n"
        )

        convo_for_model = [{"role": "user", "content": stats_context}] + st.session_state.ollama_messages
        with st.chat_message("assistant"):
            with st.spinner("Local model is thinking..."):
                reply = ask_ollama(convo_for_model, model_name="llama3")
                st.markdown(reply)
        st.session_state.ollama_messages.append({"role": "assistant", "content": reply})

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Visual Analysis", "📋 Data Table & Management", "📈 Custom Graphs"])

# Tab 1
with tab1:
    st.header("Visual Spending Analysis")
    st.subheader("Spending by Category")
    category_sums = df.groupby("category")["amount_paid"].sum().reset_index()
    fig_cat = px.pie(
        category_sums,
        values="amount_paid",
        names="category",
        title="Percentage Breakdown of Expenses",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Tealgrn,
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")
    st.subheader("Monthly Spending Trend")
    monthly_trend = df.groupby("YearMonth")["amount_paid"].sum().reset_index()
    fig_trend = px.line(
        monthly_trend,
        x="YearMonth",
        y="amount_paid",
        markers=True,
        title="Total Spending Over Time",
        labels={"YearMonth": "Month", "amount_paid": "Total Spent (₹)"},
        line_shape="spline",
    )
    fig_trend.update_xaxes(tickangle=45)
    st.plotly_chart(fig_trend, use_container_width=True)

# Tab 2
with tab2:
    st.header("Expense Data and Management")
    st.subheader("Full Expense History")
    display_df = df[["id", "date", "category", "amount_paid", "description", "payment_mode", "cashback"]].copy()
    display_df.columns = ["ID", "Date", "Category", "Amount", "Description", "Payment Mode", "Cashback"]
    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")
    st.subheader("🗑 Delete Expense")
    with st.expander("Click here to delete an expense by ID"):
        id_map = display_df.set_index("ID").apply(
            lambda row: f"ID {row.name}: {row['Date']} - ₹{row['Amount']} ({row['Category']})",
            axis=1,
        ).to_dict()
        ids = list(id_map.keys())
        summaries = list(id_map.values())
        if ids:
            selected_summary = st.selectbox(
                "Select the expense to delete:",
                summaries,
                index=0,
            )
            try:
                selected_id = int(selected_summary.split(":")[0].split(" ")[1])
            except (IndexError, ValueError):
                st.error("Error parsing expense ID.")
                selected_id = None

            if selected_id is not None:
                st.warning(f"Are you sure you want to delete expense ID: **{selected_id}**?")
                if st.button("Confirm Delete"):
                    delete_expense(selected_id)
                    st.success(f"❌ Expense ID {selected_id} deleted successfully. Refreshing...")
                    st.rerun()
        else:
            st.info("No expenses to delete.")

    st.markdown("---")
    st.download_button(
        label="⬇ Download All Expenses CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="expense_data.csv",
        mime="text/csv",
    )

# Tab 3 – Custom graphs
with tab3:
    st.header("Custom Expense Graphs")
    df_custom = df.copy()
    df_custom["date"] = pd.to_datetime(df_custom["date"])
    graph_type = st.radio(
        "Select graph type",
        ["📅 Monthly Expense", "🗓 Weekly Expense", "🏷 Category-wise for a Month"],
        horizontal=True,
    )

    if graph_type == "📅 Monthly Expense":
        st.subheader("Monthly Expense Trend")
        monthly = (
            df_custom.set_index("date")
            .resample("ME")["amount_paid"]
            .sum()
            .reset_index()
        )
        if monthly.empty:
            st.info("No data to display.")
        else:
            monthly["Month"] = monthly["date"].dt.strftime("%Y-%m")
            fig = px.line(
                monthly,
                x="Month",
                y="amount_paid",
                markers=True,
                title="Monthly Total Spending",
                labels={"amount_paid": "Total Spent (₹)", "Month": "Month"},
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "🗓 Weekly Expense":
        st.subheader("Weekly Expense Trend")
        weekly = (
            df_custom.set_index("date")
            .resample("W-MON")["amount_paid"]
            .sum()
            .reset_index()
        )
        if weekly.empty:
            st.info("No data to display.")
        else:
            weekly["Week"] = weekly["date"].dt.strftime("%Y-%m-%d")
            fig = px.bar(
                weekly,
                x="Week",
                y="amount_paid",
                title="Weekly Total Spending (Weeks starting Monday)",
                labels={"amount_paid": "Total Spent (₹)", "Week": "Week Start Date"},
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader("Category-wise Spending for a Month")
        df_custom["YearMonth"] = df_custom["date"].dt.strftime("%Y-%m")
        available_months = sorted(df_custom["YearMonth"].unique())
        selected_month = st.selectbox("Select Month", available_months)
        month_df = df_custom[df_custom["YearMonth"] == selected_month]
        if month_df.empty:
            st.info("No data for this month.")
        else:
            cat_month = month_df.groupby("category")["amount_paid"].sum().reset_index()
            fig = px.bar(
                cat_month,
                x="category",
                y="amount_paid",
                title=f"Spending by Category in {selected_month}",
                labels={"category": "Category", "amount_paid": "Total Spent (₹)"},
            )
            st.plotly_chart(fig, use_container_width=True)
