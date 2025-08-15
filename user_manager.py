import argparse
import getpass
from backend.database import SessionLocal
from backend import crud, models, security # Import models and security for hashing

# --- Command Handler Functions ---

def handle_create_user(args):
    """Handler for the 'create' command."""
    db = SessionLocal()
    print(f"--> Attempting to create user '{args.username}'")

    if crud.get_user_by_username(db, username=args.username):
        print(f"Error: User '{args.username}' already exists.")
        db.close()
        return

    try:
        password = getpass.getpass("Enter password: ")
        if not password:
            print("Error: Password cannot be empty.")
            db.close()
            return
    except Exception as e:
        print(f"Could not read password: {e}")
        db.close()
        return

    user_in = crud.UserCreate(
        username=args.username,
        password=password,
        email=args.email,
        role=args.role,
        receives_email_alerts=args.enable_alerts
    )
    
    created_user = crud.create_user(db=db, user=user_in)
    print(f"Success! User '{created_user.username}' created with role '{created_user.role.value}'.")
    db.close()

def handle_update_user(args):
    """Handler for the 'update' command."""
    db = SessionLocal()
    print(f"--> Attempting to update user '{args.username}'")

    user = crud.get_user_by_username(db, args.username)
    if not user:
        print(f"Error: User '{args.username}' not found.")
        db.close()
        return
        
    updates = {}
    if args.email: updates['email'] = args.email
    if args.role: updates['role'] = args.role
    if args.alerts_on: updates['receives_email_alerts'] = True
    if args.alerts_off: updates['receives_email_alerts'] = False
    
    if not updates:
        print("No update specified. Use --email, --role, --enable-alerts, or --disable-alerts.")
        db.close()
        return

    crud.update_user(db, user, updates)
    print(f"Successfully updated user '{args.username}'.")
    db.close()
    
def handle_update_password(args):
    """Handler for the 'update-password' command."""
    db = SessionLocal()
    print(f"--> Updating password for user '{args.username}'")

    user_to_update = crud.get_user_by_username(db, username=args.username)
    if not user_to_update:
        print(f"Error: User '{args.username}' not found.")
        db.close()
        return
    
    try:
        new_password = getpass.getpass("Enter new password: ")
        new_password_confirm = getpass.getpass("Confirm new password: ")
        if new_password != new_password_confirm:
            print("Error: Passwords do not match.")
            db.close()
            return
        if not new_password:
            print("Error: Password cannot be empty.")
            db.close()
            return
    except Exception as e:
        print(f"Could not read password: {e}")
        db.close()
        return

    crud.update_user_password(db=db, user=user_to_update, new_password=new_password)
    print(f"Success! Password for user '{args.username}' has been updated.")
    db.close()

# --- CLI Argument Parser Definition ---

def main():
    parser = argparse.ArgumentParser(
        description="PdM Application User Management CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- 'create' command ---
    parser_create = subparsers.add_parser('create', help='Create a new user')
    parser_create.add_argument('--username', required=True, help="Username for the new user.")
    parser_create.add_argument('--email', required=True, help="User's email address (for alerts).")
    parser_create.add_argument('--role', required=True, choices=[r.value for r in models.UserRole], help="User's role.")
    parser_create.add_argument('--enable-alerts', action='store_true', help="Enable email alerts for this user.")
    parser_create.set_defaults(func=handle_create_user)

    # --- 'update' command ---
    parser_update = subparsers.add_parser('update', help="Update an existing user's details (email, role, alert status).")
    parser_update.add_argument('--username', required=True, help="Username of the user to update.")
    parser_update.add_argument('--email', help="New email address.")
    parser_update.add_argument('--role', choices=[r.value for r in models.UserRole], help="New role.")
    alert_group = parser_update.add_mutually_exclusive_group()
    alert_group.add_argument('--enable-alerts', action='store_true', dest='alerts_on')
    alert_group.add_argument('--disable-alerts', action='store_true', dest='alerts_off')
    parser_update.set_defaults(func=handle_update_user)
    
    # --- 'update-password' command ---
    parser_password = subparsers.add_parser('update-password', help="Update an existing user's password securely.")
    parser_password.add_argument('--username', required=True, help="Username of the user to update.")
    parser_password.set_defaults(func=handle_update_password)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()