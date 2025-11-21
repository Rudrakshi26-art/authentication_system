import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from datetime import datetime, timedelta
import logging
from utils import Utils

class AttendanceViewerWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Attendance Records")
        self.window.geometry("1000x600")
        self.window.configure(bg='#2c3e50')
        
        self.df = Utils.load_attendance_records()
        self.setup_ui()
        self.load_data()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Filter frame
        filter_frame = ttk.LabelFrame(main_frame, text="Filters", padding=10)
        filter_frame.pack(fill='x', pady=(0, 10))
        
        # Date filter
        ttk.Label(filter_frame, text="Date:").grid(row=0, column=0, sticky='w', padx=(0, 5))
        self.date_var = tk.StringVar()
        date_combo = ttk.Combobox(filter_frame, textvariable=self.date_var, state='readonly')
        date_combo['values'] = self.get_date_options()
        date_combo.grid(row=0, column=1, sticky='w', padx=(0, 20))
        date_combo.bind('<<ComboboxSelected>>', self.apply_filters)
        
        # Employee ID filter
        ttk.Label(filter_frame, text="Employee ID:").grid(row=0, column=2, sticky='w', padx=(0, 5))
        self.emp_id_var = tk.StringVar()
        emp_combo = ttk.Combobox(filter_frame, textvariable=self.emp_id_var)
        emp_combo['values'] = self.get_employee_ids()
        emp_combo.grid(row=0, column=3, sticky='w', padx=(0, 20))
        emp_combo.bind('<<ComboboxSelected>>', self.apply_filters)
        
        # Department filter
        ttk.Label(filter_frame, text="Department:").grid(row=0, column=4, sticky='w', padx=(0, 5))
        self.dept_var = tk.StringVar()
        dept_combo = ttk.Combobox(filter_frame, textvariable=self.dept_var)
        dept_combo['values'] = self.get_departments()
        dept_combo.grid(row=0, column=5, sticky='w', padx=(0, 20))
        dept_combo.bind('<<ComboboxSelected>>', self.apply_filters)
        
        # Clear filters button
        ttk.Button(filter_frame, text="Clear Filters", 
                  command=self.clear_filters).grid(row=0, column=6, padx=(20, 0))
        
        # Statistics frame
        stats_frame = ttk.Frame(main_frame)
        stats_frame.pack(fill='x', pady=(0, 10))
        
        self.stats_label = ttk.Label(stats_frame, text="", font=('Arial', 10))
        self.stats_label.pack(anchor='w')
        
        # Treeview frame
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill='both', expand=True)
        
        # Create treeview
        columns = ('employee_id', 'name', 'department', 'date', 'time')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings')
        
        # Define headings
        self.tree.heading('employee_id', text='Employee ID')
        self.tree.heading('name', text='Name')
        self.tree.heading('department', text='Department')
        self.tree.heading('date', text='Date')
        self.tree.heading('time', text='Time')
        
        # Set column widths
        self.tree.column('employee_id', width=100)
        self.tree.column('name', width=150)
        self.tree.column('department', width=120)
        self.tree.column('date', width=100)
        self.tree.column('time', width=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Export button
        ttk.Button(main_frame, text="Export to CSV", 
                  command=self.export_csv).pack(side='right', pady=(10, 0))
    
    def get_date_options(self):
        dates = ['All Dates'] + sorted(self.df['date'].unique().tolist(), reverse=True)
        return dates
    
    def get_employee_ids(self):
        return ['All Employees'] + sorted(self.df['employee_id'].unique().tolist())
    
    def get_departments(self):
        return ['All Departments'] + sorted(self.df['department'].unique().tolist())
    
    def load_data(self):
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Populate treeview
        for _, row in self.df.iterrows():
            self.tree.insert('', 'end', values=(
                row['employee_id'],
                row['name'],
                row['department'],
                row['date'],
                row['time']
            ))
        
        # Update statistics
        total_records = len(self.df)
        unique_employees = self.df['employee_id'].nunique()
        today = datetime.now().strftime("%Y-%m-%d")
        today_count = len(self.df[self.df['date'] == today])
        
        stats_text = f"Total Records: {total_records} | Unique Employees: {unique_employees} | Today's Attendance: {today_count}"
        self.stats_label.configure(text=stats_text)
    
    def apply_filters(self, event=None):
        filtered_df = self.df.copy()
        
        # Apply date filter
        if self.date_var.get() and self.date_var.get() != 'All Dates':
            filtered_df = filtered_df[filtered_df['date'] == self.date_var.get()]
        
        # Apply employee ID filter
        if self.emp_id_var.get() and self.emp_id_var.get() != 'All Employees':
            filtered_df = filtered_df[filtered_df['employee_id'] == self.emp_id_var.get()]
        
        # Apply department filter
        if self.dept_var.get() and self.dept_var.get() != 'All Departments':
            filtered_df = filtered_df[filtered_df['department'] == self.dept_var.get()]
        
        # Update treeview
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for _, row in filtered_df.iterrows():
            self.tree.insert('', 'end', values=(
                row['employee_id'],
                row['name'],
                row['department'],
                row['date'],
                row['time']
            ))
    
    def clear_filters(self):
        self.date_var.set('')
        self.emp_id_var.set('')
        self.dept_var.set('')
        self.load_data()
    
    def export_csv(self):
        try:
            filename = f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.df.to_csv(filename, index=False)
            messagebox.showinfo("Success", f"Data exported to {filename}")
        except Exception as e:
            logging.error(f"Export error: {str(e)}")
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
