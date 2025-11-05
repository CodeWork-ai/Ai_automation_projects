import sqlite3
from sqlite_utils import Database
import threading

# Thread-local storage for database connections
_thread_local = threading.local()

def get_db():
    """
    Creates and returns a thread-safe database connection.
    Each thread gets its own connection.
    """
    if not hasattr(_thread_local, 'db'):
        conn = sqlite3.connect("assistant.db", check_same_thread=False)
        _thread_local.db = Database(conn)
        
        # Create tables if they don't exist
        if "crunchbase" not in _thread_local.db.table_names():
            _thread_local.db["crunchbase"].create({
                "url": str,
                "name": str,
                "description": str,
                "data": str
            }, pk="url")

        if "amazon_products" not in _thread_local.db.table_names():
            _thread_local.db["amazon_products"].create({
                "url": str,
                "data": str,
            }, pk="url")
        
        # Add the new table for Yahoo Finance data
        if "yahoo_finance" not in _thread_local.db.table_names():
            _thread_local.db["yahoo_finance"].create({
                "url": str,
                "name": str,
                "description": str,
                "data": str
            }, pk="url")
    
    return _thread_local.db