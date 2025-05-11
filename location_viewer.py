import os
import json
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QListWidget, 
                            QFileDialog, QMessageBox, QSlider, QCheckBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWebEngineWidgets import QWebEngineView
from folium import Map, Marker, CircleMarker, Popup, LayerControl
from folium.plugins import MarkerCluster, HeatMap
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import webbrowser
import tempfile

class LocationViewer(QMainWindow):
    def __init__(self, data_dir='collected_data'):
        super().__init__()
        self.setWindowTitle("NoiseNinja - Location Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        self.data_dir = data_dir
        self.recordings = self._load_recordings()
        self.current_map = None
        self.temp_map_file = None
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        
        # Class filter
        self.class_label = QLabel("Filter by Class:")
        self.class_selector = QComboBox()
        self.class_selector.addItems(self._get_classes())
        self.class_selector.currentTextChanged.connect(self.update_map)
        
        # Date range filter
        self.date_label = QLabel("Date Range:")
        self.date_selector = QComboBox()
        self.date_selector.addItems(["All Time", "Last 24 Hours", "Last Week", "Last Month"])
        self.date_selector.currentTextChanged.connect(self.update_map)
        
        # Clustering controls
        self.cluster_label = QLabel("Clustering:")
        self.cluster_checkbox = QCheckBox("Enable Clustering")
        self.cluster_checkbox.stateChanged.connect(self.update_map)
        
        self.distance_label = QLabel("Cluster Distance (meters):")
        self.distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.distance_slider.setRange(10, 1000)
        self.distance_slider.setValue(100)
        self.distance_slider.valueChanged.connect(self.update_map)
        
        # Export controls
        self.export_button = QPushButton("Export Selected")
        self.export_button.clicked.connect(self.export_selected)
        
        # Recording list
        self.recording_list = QListWidget()
        self.recording_list.itemClicked.connect(self.highlight_recording)
        
        # Add widgets to control panel
        control_layout.addWidget(self.class_label)
        control_layout.addWidget(self.class_selector)
        control_layout.addWidget(self.date_label)
        control_layout.addWidget(self.date_selector)
        control_layout.addWidget(self.cluster_label)
        control_layout.addWidget(self.cluster_checkbox)
        control_layout.addWidget(self.distance_label)
        control_layout.addWidget(self.distance_slider)
        control_layout.addWidget(self.export_button)
        control_layout.addWidget(QLabel("Recordings:"))
        control_layout.addWidget(self.recording_list)
        
        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(300)
        
        # Create map view
        self.map_view = QWebEngineView()
        
        # Add widgets to main layout
        layout.addWidget(control_panel)
        layout.addWidget(self.map_view)
        
        main_widget.setLayout(layout)
        
        # Initial map update
        self.update_map()
    
    def _load_recordings(self):
        """Load all recordings and their metadata"""
        recordings = []
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.endswith('.json'):
                        with open(os.path.join(class_path, file), 'r') as f:
                            metadata = json.load(f)
                            if metadata.get('location'):
                                recordings.append(metadata)
        return recordings
    
    def _get_classes(self):
        """Get list of all sound classes"""
        return list(set(r['class'] for r in self.recordings))
    
    def _filter_recordings(self):
        """Filter recordings based on current selection"""
        filtered = self.recordings.copy()
        
        # Filter by class
        selected_class = self.class_selector.currentText()
        if selected_class != "All":
            filtered = [r for r in filtered if r['class'] == selected_class]
        
        # Filter by date
        date_range = self.date_selector.currentText()
        if date_range != "All Time":
            now = datetime.now()
            if date_range == "Last 24 Hours":
                cutoff = now.timestamp() - 86400
            elif date_range == "Last Week":
                cutoff = now.timestamp() - 604800
            elif date_range == "Last Month":
                cutoff = now.timestamp() - 2592000
            
            filtered = [r for r in filtered if datetime.fromisoformat(r['timestamp']).timestamp() > cutoff]
        
        return filtered
    
    def _create_clusters(self, recordings):
        """Create clusters of nearby recordings"""
        if not recordings:
            return []
        
        # Extract coordinates
        coords = np.array([[r['location']['latitude'], r['location']['longitude']] for r in recordings])
        
        # Convert distance to degrees (approximate)
        distance_degrees = self.distance_slider.value() / 111000  # 111km per degree
        
        # Perform clustering
        clustering = DBSCAN(eps=distance_degrees, min_samples=2).fit(coords)
        
        # Group recordings by cluster
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(recordings[idx])
        
        return clusters
    
    def update_map(self):
        """Update the map with current filters and clustering"""
        filtered_recordings = self._filter_recordings()
        
        # Create map centered on first recording or default location
        if filtered_recordings:
            center_lat = filtered_recordings[0]['location']['latitude']
            center_lon = filtered_recordings[0]['location']['longitude']
        else:
            center_lat, center_lon = 0, 0
        
        m = Map(location=[center_lat, center_lon], zoom_start=13)
        
        if self.cluster_checkbox.isChecked():
            # Create clusters
            clusters = self._create_clusters(filtered_recordings)
            
            # Add cluster markers
            for cluster_id, recordings in clusters.items():
                if cluster_id == -1:  # Noise points
                    for recording in recordings:
                        self._add_recording_marker(m, recording)
                else:
                    # Calculate cluster center
                    center_lat = np.mean([r['location']['latitude'] for r in recordings])
                    center_lon = np.mean([r['location']['longitude'] for r in recordings])
                    
                    # Create cluster marker
                    popup_content = f"<b>Cluster {cluster_id}</b><br>"
                    popup_content += f"Number of recordings: {len(recordings)}<br>"
                    popup_content += f"Classes: {', '.join(set(r['class'] for r in recordings))}"
                    
                    CircleMarker(
                        location=[center_lat, center_lon],
                        radius=10,
                        popup=Popup(popup_content),
                        color='red',
                        fill=True
                    ).add_to(m)
        else:
            # Add individual markers
            for recording in filtered_recordings:
                self._add_recording_marker(m, recording)
        
        # Save map to temporary file
        if self.temp_map_file:
            os.remove(self.temp_map_file)
        self.temp_map_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html').name
        m.save(self.temp_map_file)
        
        # Update map view
        self.map_view.setUrl(Qt.QUrl.fromLocalFile(self.temp_map_file))
        
        # Update recording list
        self.recording_list.clear()
        for recording in filtered_recordings:
            self.recording_list.addItem(f"{recording['class']} - {recording['timestamp']}")
    
    def _add_recording_marker(self, map_obj, recording):
        """Add a marker for a single recording"""
        popup_content = f"<b>{recording['class']}</b><br>"
        popup_content += f"Time: {recording['timestamp']}<br>"
        popup_content += f"Duration: {recording['duration']:.1f}s"
        
        if recording['location'].get('address'):
            popup_content += f"<br>Address: {recording['location']['address']}"
        
        Marker(
            location=[recording['location']['latitude'], recording['location']['longitude']],
            popup=Popup(popup_content)
        ).add_to(map_obj)
    
    def highlight_recording(self, item):
        """Highlight a recording on the map"""
        idx = self.recording_list.row(item)
        filtered_recordings = self._filter_recordings()
        if 0 <= idx < len(filtered_recordings):
            recording = filtered_recordings[idx]
            self.map_view.setUrl(Qt.QUrl.fromLocalFile(self.temp_map_file))
            # Center map on selected recording
            js = f"map.setView([{recording['location']['latitude']}, {recording['location']['longitude']}], 15);"
            self.map_view.page().runJavaScript(js)
    
    def export_selected(self):
        """Export selected recordings"""
        selected_items = self.recording_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select recordings to export")
            return
        
        # Get export directory
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return
        
        # Export selected recordings
        filtered_recordings = self._filter_recordings()
        for item in selected_items:
            idx = self.recording_list.row(item)
            if 0 <= idx < len(filtered_recordings):
                recording = filtered_recordings[idx]
                
                # Create export directory structure
                class_dir = os.path.join(export_dir, recording['class'])
                os.makedirs(class_dir, exist_ok=True)
                
                # Copy audio file
                audio_path = os.path.join(self.data_dir, recording['class'], 
                                        recording['filename'])
                if os.path.exists(audio_path):
                    import shutil
                    shutil.copy2(audio_path, os.path.join(class_dir, recording['filename']))
                
                # Copy metadata
                metadata_path = audio_path.replace('.wav', '.json')
                if os.path.exists(metadata_path):
                    shutil.copy2(metadata_path, 
                               os.path.join(class_dir, recording['filename'].replace('.wav', '.json')))
        
        QMessageBox.information(self, "Success", "Selected recordings exported successfully")

def main():
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = LocationViewer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 