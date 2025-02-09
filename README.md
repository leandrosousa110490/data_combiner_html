# Data File Converter

A modern, user-friendly Flask web application for converting various data file formats to CSV. This application supports large datasets and provides real-time progress tracking and data statistics.

## Features

- 📁 Support for multiple file formats:
  - CSV
  - Excel (XLSX, XLS)
  - JSON
  - Parquet
  - Text files
- 🔄 Interactive drag-and-drop file upload
- 📊 Real-time progress tracking
- 📈 Data statistics display
- 🎨 Modern, responsive UI with Bootstrap
- ⚙️ Custom separator options for CSV/TXT files
- 💾 Automatic file conversion to CSV
- 📱 Mobile-friendly design

## Requirements

- Python 3.8+
- Flask
- Pandas
- NumPy
- Other dependencies (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd data-file-converter
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the application:
   - Drag and drop your file or click to browse
   - Select separator options for CSV/TXT files
   - Monitor upload and conversion progress
   - View file statistics
   - Download the converted CSV file

## Project Structure

```
data-file-converter/
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css  # Custom styles
│   └── js/
│       └── main.js    # Frontend JavaScript
├── templates/
│   └── index.html     # Main HTML template
└── uploads/           # Temporary file storage
```

## Features in Detail

### File Upload
- Drag and drop interface
- Progress bar for upload tracking
- File type validation
- File size display

### Data Processing
- Automatic format detection
- Custom separator selection
- Large file handling
- Memory-efficient processing

### User Interface
- Bootstrap-based responsive design
- Interactive elements
- Real-time feedback
- Mobile-friendly layout

### Data Statistics
- Row count
- Column count
- Memory usage
- File information

## Security Features

- Secure file handling
- Input validation
- File size limits
- Temporary file cleanup

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 