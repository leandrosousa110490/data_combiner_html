from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, session
import pandas as pd
import polars as pl
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
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow as pa
import pyarrow.csv as csv
import duckdb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize DuckDB with optimizations
try:
    # Create a DuckDB connection
    db_conn = duckdb.connect()
    db_conn.execute("SET enable_progress_bar=true")
    db_conn.execute("SET memory_limit='16GB'")
    db_conn.execute("SET temp_directory='uploads/temp'")
except Exception as e:
    logger.error(f"Error initializing DuckDB: {str(e)}")
    db_conn = duckdb.connect()  # Fallback with default settings

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
# Increase max content length to 10GB per file
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024

# Create uploads directory with subdirectories for better organization
for subdir in ['input', 'output', 'temp']:
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], subdir), exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'parquet', 'txt'}
CHUNK_SIZE = 500000  # Increased chunk size for better performance
MAX_WORKERS = max(1, psutil.cpu_count() - 1)  # Leave one CPU core free

# Store file processing status
processing_status = {}

def optimize_numeric_array(arr):
    """Optimize numeric array operations"""
    return np.nan_to_num(arr, 0)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_files(file_id, input_files, output_file):
    """Clean up input and output files"""
    try:
        # Clean up input files
        for filepath in input_files:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Cleaned up input file: {filepath}")
        
        # Clean up output file
        if os.path.exists(output_file):
            os.remove(output_file)
            logger.info(f"Cleaned up output file: {output_file}")
        
        # Remove from processing status
        if file_id in processing_status:
            del processing_status[file_id]
            
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def get_memory_usage():
    """Get current memory usage in percentage"""
    return psutil.Process().memory_percent()

def adjust_chunk_size():
    """Dynamically adjust chunk size based on available memory"""
    memory_usage = get_memory_usage()
    if memory_usage > 75:
        return CHUNK_SIZE // 2
    elif memory_usage < 25:
        return CHUNK_SIZE * 2
    return CHUNK_SIZE

def process_chunk(chunk, all_columns):
    """Process a single chunk of data with optimized operations"""
    try:
        if isinstance(chunk, pl.LazyFrame):
            # Handle Polars LazyFrame
            df = chunk.collect().to_pandas()
        else:
            df = chunk

        # Create DataFrame with all columns
        df_reindexed = pd.DataFrame(index=df.index, columns=all_columns)
        
        # Process columns in batches for better performance
        for col in all_columns:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df_reindexed[col] = np.nan_to_num(df[col].values, 0)
                else:
                    df_reindexed[col] = df[col].fillna('')
            else:
                if pd.api.types.is_numeric_dtype(df_reindexed[col].dtype):
                    df_reindexed[col] = 0
                else:
                    df_reindexed[col] = ''
        
        return df_reindexed
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return None

def read_file_in_chunks(filepath, ext, separator=',', chunksize=None):
    """Read file in chunks with optimized settings for large files"""
    if chunksize is None:
        chunksize = adjust_chunk_size()
    
    try:
        if ext == 'csv' or ext == 'txt':
            # Use Polars for faster CSV reading
            return pl.scan_csv(filepath, separator=separator).with_streaming(True)
        elif ext in ['xlsx', 'xls']:
            # Use openpyxl for Excel files with memory optimization
            try:
                # First try reading with pandas directly
                df = pd.read_excel(filepath, engine='openpyxl')
                return [df]
            except Exception as e:
                logger.warning(f"Memory-efficient reading for {filepath}: {str(e)}")
                # If that fails, try reading in chunks manually
                import openpyxl
                wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
                sheet = wb.active
                
                # Get headers
                headers = [cell.value for cell in next(sheet.rows)]
                
                # Process rows in chunks
                chunk_data = []
                current_chunk = []
                
                for row in sheet.rows:
                    if len(current_chunk) >= chunksize:
                        chunk_df = pd.DataFrame(current_chunk, columns=headers)
                        chunk_data.append(chunk_df)
                        current_chunk = []
                    
                    row_data = [cell.value for cell in row]
                    current_chunk.append(row_data)
                
                # Add the last chunk if any
                if current_chunk:
                    chunk_df = pd.DataFrame(current_chunk, columns=headers)
                    chunk_data.append(chunk_df)
                
                wb.close()
                return chunk_data
                
        elif ext == 'json':
            # Use Polars for JSON files
            return pl.scan_ndjson(filepath).with_streaming(True)
        elif ext == 'parquet':
            # Use DuckDB for Parquet files with memory optimizations
            query = f"""
            SELECT * FROM parquet_scan('{filepath}')
            """
            return db_conn.execute(query).df()
        return None
    except Exception as e:
        raise Exception(f"Error reading file {filepath}: {str(e)}")

def read_sample_file(filepath, ext, separator):
    """Read the first row of a file with optimized readers"""
    try:
        if ext in ['csv', 'txt']:
            return pl.read_csv(filepath, separator=separator, n_rows=1).to_pandas()
        elif ext in ['xlsx', 'xls']:
            # Use pandas to read first row of Excel files
            return pd.read_excel(filepath, nrows=1)
        elif ext == 'json':
            return pl.read_ndjson(filepath, n_rows=1).to_pandas()
        elif ext == 'parquet':
            query = f"""
            SELECT * FROM parquet_scan('{filepath}') LIMIT 1
            """
            return db_conn.execute(query).df()
        return None
    except Exception as e:
        raise Exception(f"Error reading sample from {filepath}: {str(e)}")

def process_files(file_id, filepaths, output_path, separator=','):
    try:
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
            processing_status[file_id]['progress'] = (i / total_files) * 30
            
            try:
                ext = filepath.rsplit('.', 1)[1].lower()
                df_sample = read_sample_file(filepath, ext, separator)
                if df_sample is not None:
                    all_columns.update(df_sample.columns)
            except Exception as e:
                processing_status[file_id]['errors'].append(f"Warning: Could not read columns from {os.path.basename(filepath)}: {str(e)}")
        
        # Convert to list and sort for consistent column order
        all_columns = sorted(list(all_columns))
        
        # Process each file
        total_rows = 0
        processed_files = []
        
        # Create output file with PyArrow for better performance
        schema = pa.schema([pa.field(col, pa.string()) for col in all_columns])
        writer = csv.CSVWriter(output_path, schema)
        
        for i, filepath in enumerate(filepaths):
            try:
                processing_status[file_id]['current_operation'] = f'Processing file {i+1} of {total_files}'
                base_progress = 30 + (i / total_files) * 70
                
                ext = filepath.rsplit('.', 1)[1].lower()
                chunks = read_file_in_chunks(filepath, ext, separator)
                
                if isinstance(chunks, pl.LazyFrame):
                    # Process Polars LazyFrame in chunks
                    chunk_count = 0
                    for chunk in chunks.collect(streaming=True):
                        processed_chunk = process_chunk(chunk, all_columns)
                        if processed_chunk is not None:
                            writer.write_table(pa.Table.from_pandas(processed_chunk))
                            total_rows += len(processed_chunk)
                        chunk_count += 1
                        
                        # Update progress
                        if chunk_count % 10 == 0:
                            chunk_progress = base_progress + (chunk_count / (chunk_count + 1)) * (70 / total_files)
                            processing_status[file_id]['progress'] = min(99, chunk_progress)
                            gc.collect()
                
                elif isinstance(chunks, pd.DataFrame):
                    # Process DuckDB result
                    processed_chunk = process_chunk(chunks, all_columns)
                    if processed_chunk is not None:
                        writer.write_table(pa.Table.from_pandas(processed_chunk))
                        total_rows += len(processed_chunk)
                
                else:  # Handle regular chunks (e.g., from Excel files)
                    # Use ThreadPoolExecutor for parallel processing
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = [executor.submit(process_chunk, chunk, all_columns) for chunk in chunks]
                        for future in as_completed(futures):
                            processed_chunk = future.result()
                            if processed_chunk is not None:
                                writer.write_table(pa.Table.from_pandas(processed_chunk))
                                total_rows += len(processed_chunk)
                
                # Mark file as processed and delete it
                processed_files.append(filepath)
                os.remove(filepath)
                logger.info(f"Processed and deleted: {filepath}")
                
            except Exception as e:
                processing_status[file_id]['errors'].append(f"Error processing {os.path.basename(filepath)}: {str(e)}")
        
        # Close the writer
        writer.close()
        
        # Update status with completion and statistics
        processing_status[file_id].update({
            'status': 'completed',
            'progress': 100,
            'stats': {
                'total_files_processed': total_files,
                'total_rows': total_rows,
                'columns': len(all_columns),
                'column_names': all_columns,
                'output_size': f"{os.path.getsize(output_path) / (1024 * 1024 * 1024):.2f} GB"
            }
        })

    except Exception as e:
        logger.error(f"Error in process_files: {str(e)}")
        processing_status[file_id].update({
            'status': 'error',
            'error': str(e)
        })
        # Clean up any remaining input files
        for filepath in filepaths:
            if filepath not in processed_files and os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Deleted input file after error: {filepath}")

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