from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, session
import pandas as pd
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import uuid
import threading
import time
from pathlib import Path
import gc
import numpy as np
from tqdm import tqdm
import shutil

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages and session
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size per file

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'parquet', 'txt'}

# Store file processing status
processing_status = {}

def cleanup_files(file_id, filepaths, output_path, cleanup_output=True):
    """Clean up uploaded and processed files"""
    try:
        # Delete all uploaded files
        for filepath in filepaths:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Deleted input file: {filepath}")
        
        # Delete the output file only if specified
        if cleanup_output and os.path.exists(output_path):
            os.remove(output_path)
            print(f"Deleted output file: {output_path}")
        
        # Remove the processing status if cleaning up everything
        if cleanup_output and file_id in processing_status:
            del processing_status[file_id]
            
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_file_in_chunks(filepath, ext, separator=',', chunksize=10000):
    """Read file in chunks to handle large files efficiently"""
    try:
        if ext == 'csv' or ext == 'txt':
            return pd.read_csv(filepath, sep=separator, chunksize=chunksize)
        elif ext in ['xlsx', 'xls']:
            # Excel files need to be read completely, but we'll process them in chunks
            df = pd.read_excel(filepath)
            return np.array_split(df, max(1, len(df) // chunksize))
        elif ext == 'json':
            # For JSON, we'll read in chunks using lines=True
            return pd.read_json(filepath, lines=True, chunksize=chunksize)
        elif ext == 'parquet':
            # Parquet files are already optimized for chunked reading
            return pd.read_parquet(filepath, engine='fastparquet')
        return None
    except Exception as e:
        raise Exception(f"Error reading file {filepath}: {str(e)}")

def process_files(file_id, filepaths, output_path, separator=','):
    try:
        # Update status to processing
        processing_status[file_id]['status'] = 'processing'
        processing_status[file_id]['progress'] = 0
        total_files = len(filepaths)
        
        # Store filepaths for cleanup
        processing_status[file_id]['input_files'] = filepaths
        processing_status[file_id]['output_file'] = output_path
        
        # Create empty list to store column names from all files
        all_columns = set()
        
        # First pass: collect all unique column names
        for i, filepath in enumerate(filepaths):
            processing_status[file_id]['current_operation'] = f'Analyzing file {i+1} of {total_files}'
            processing_status[file_id]['progress'] = (i / total_files) * 30  # Use first 30% for analysis
            
            ext = filepath.rsplit('.', 1)[1].lower()
            try:
                # Read just the header of each file
                if ext in ['csv', 'txt']:
                    df_sample = pd.read_csv(filepath, sep=separator, nrows=1)
                elif ext in ['xlsx', 'xls']:
                    df_sample = pd.read_excel(filepath, nrows=1)
                elif ext == 'json':
                    df_sample = pd.read_json(filepath, lines=True, nrows=1)
                elif ext == 'parquet':
                    df_sample = pd.read_parquet(filepath, engine='fastparquet').head(1)
                
                all_columns.update(df_sample.columns)
            except Exception as e:
                processing_status[file_id]['errors'].append(f"Warning: Could not read columns from {os.path.basename(filepath)}: {str(e)}")
        
        # Convert to list and sort for consistent column order
        all_columns = sorted(list(all_columns))
        
        # Initialize the output file with headers
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(','.join(all_columns) + '\n')
        
        # Process each file
        total_rows = 0
        processed_files = []  # Keep track of successfully processed files
        
        for i, filepath in enumerate(filepaths):
            processing_status[file_id]['current_operation'] = f'Processing file {i+1} of {total_files}'
            base_progress = 30 + (i / total_files) * 70  # Remaining 70% for processing
            
            ext = filepath.rsplit('.', 1)[1].lower()
            try:
                chunks = read_file_in_chunks(filepath, ext, separator)
                chunk_count = 0
                
                for chunk in chunks:
                    # Reindex the chunk to include all columns
                    chunk_reindexed = chunk.reindex(columns=all_columns)
                    
                    # Append to CSV without headers (except for first chunk)
                    chunk_reindexed.to_csv(output_path, mode='a', header=False, index=False)
                    
                    total_rows += len(chunk_reindexed)
                    chunk_count += 1
                    
                    # Update progress
                    chunk_progress = base_progress + (chunk_count / (chunk_count + 1)) * (70 / total_files)
                    processing_status[file_id]['progress'] = min(99, chunk_progress)
                    
                    # Force garbage collection after each chunk
                    del chunk_reindexed
                    gc.collect()
                
                # Add to processed files list
                processed_files.append(filepath)
                
                # Delete the input file immediately after successful processing
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"Deleted processed file: {filepath}")
                
            except Exception as e:
                processing_status[file_id]['errors'].append(f"Error processing {os.path.basename(filepath)}: {str(e)}")
        
        # Update status with completion and statistics
        processing_status[file_id].update({
            'status': 'completed',
            'progress': 100,
            'stats': {
                'total_files_processed': total_files,
                'total_rows': total_rows,
                'columns': len(all_columns),
                'column_names': all_columns,
                'output_size': f"{os.path.getsize(output_path) / (1024 * 1024):.2f} MB"
            }
        })

    except Exception as e:
        processing_status[file_id].update({
            'status': 'error',
            'error': str(e)
        })
        # Clean up any remaining input files in case of error
        for filepath in filepaths:
            if filepath not in processed_files and os.path.exists(filepath):
                os.remove(filepath)
                print(f"Deleted input file after error: {filepath}")

@app.route('/')
def index():
    # Get any active file processing status from session
    file_id = session.get('file_id')
    status = processing_status.get(file_id, {})
    return render_template('index.html', status=status)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('index'))
    
    files = request.files.getlist('files[]')
    if not files or all(file.filename == '' for file in files):
        flash('No files selected', 'error')
        return redirect(url_for('index'))
    
    # Generate unique ID for this upload batch
    file_id = str(uuid.uuid4())
    session['file_id'] = file_id
    
    # Get separator value
    separator = request.form.get('separator', ',')
    if separator == 'custom':
        separator = request.form.get('custom_separator', ',')
    
    # Save all files and collect their paths
    filepaths = []
    total_size = 0
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
            file.save(filepath)
            filepaths.append(filepath)
            total_size += os.path.getsize(filepath)
    
    if not filepaths:
        flash('No valid files uploaded', 'error')
        return redirect(url_for('index'))
    
    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'merged_{timestamp}.csv'
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    # Initialize processing status
    processing_status[file_id] = {
        'status': 'starting',
        'progress': 0,
        'current_operation': 'Initializing...',
        'file_count': len(filepaths),
        'total_size': total_size,
        'output_filename': output_filename,
        'errors': []
    }
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_files,
        args=(file_id, filepaths, output_path, separator)
    )
    thread.start()
    
    return redirect(url_for('index'))

@app.route('/status/<file_id>')
def get_status(file_id):
    status = processing_status.get(file_id, {})
    if status.get('status') == 'completed':
        if status.get('errors'):
            flash('Processing completed with some errors. Check the error log.', 'warning')
        else:
            flash('Files merged successfully!', 'success')
    elif status.get('status') == 'error':
        flash(f'Error processing files: {status.get("error")}', 'error')
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        flash('File not found or already downloaded', 'error')
        return redirect(url_for('index'))
    
    # Get the file_id from the session
    file_id = session.get('file_id')
    
    try:
        # Send the file
        response = send_file(
            file_path,
            as_attachment=True,
            download_name=filename
        )
        
        # Schedule cleanup of only the output file after sending
        @response.call_on_close
        def cleanup():
            if file_id and file_id in processing_status:
                status = processing_status[file_id]
                if status.get('output_file'):
                    if os.path.exists(status['output_file']):
                        os.remove(status['output_file'])
                        print(f"Deleted output file after download: {status['output_file']}")
                    if file_id in processing_status:
                        del processing_status[file_id]
        
        return response
        
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('index'))

# Add a cleanup route for manual cleanup if needed
@app.route('/cleanup/<file_id>')
def cleanup_route(file_id):
    if file_id in processing_status:
        status = processing_status[file_id]
        if status.get('input_files') and status.get('output_file'):
            cleanup_files(file_id, status['input_files'], status['output_file'])
            flash('Files cleaned up successfully', 'success')
    return redirect(url_for('index'))

# Add periodic cleanup function
def cleanup_old_files():
    """Clean up files older than 1 hour"""
    try:
        current_time = time.time()
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Check if file is older than 1 hour
            if os.path.getctime(filepath) < (current_time - 3600):
                os.remove(filepath)
    except Exception as e:
        print(f"Error during periodic cleanup: {str(e)}")

# Start periodic cleanup in a background thread
def start_cleanup_thread():
    while True:
        cleanup_old_files()
        time.sleep(3600)  # Run every hour

cleanup_thread = threading.Thread(target=start_cleanup_thread, daemon=True)
cleanup_thread.start()

@app.template_filter('format_size')
def format_size(size):
    # Convert bytes to human readable format
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

if __name__ == '__main__':
    app.run(debug=True) 