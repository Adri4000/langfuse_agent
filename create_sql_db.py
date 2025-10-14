import sqlite3

# Connect (creates file if not exists)
conn = sqlite3.connect("data/sale_database.db")
cur = conn.cursor()


# Create a table
cur.execute("""
CREATE TABLE IF NOT EXISTS sales (
    product TEXT,
    region TEXT,
    date TEXT,
    sales INTEGER
)
""")

# Insert sample data
cur.execute("INSERT INTO sales (product, region, date, sales) VALUES (?, ?, ?, ?)",
            ("Product X", "North", "2025-09-01", 1000))
cur.execute("INSERT INTO sales (product, region, date, sales) VALUES (?, ?, ?, ?)",
            ("Product X", "South", "2025-09-01", 2000))

conn.commit()
conn.close()

