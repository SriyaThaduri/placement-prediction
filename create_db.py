from app import app, db

# Create all tables within the application context
with app.app_context():
    db.create_all()
    print("Database and tables created successfully.")