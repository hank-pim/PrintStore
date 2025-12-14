import sys
import os
import hashlib
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication,
    QFileIconProvider,
    QFileDialog,
    QHBoxLayout,
    QListView,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTreeView,
    QToolButton,
    QWidget,
    QVBoxLayout,
    QLabel,
    QSlider,
)
from PyQt6.QtCore import QDir, QSize, Qt, QSortFilterProxyModel
from PyQt6.QtGui import QFileSystemModel, QIcon
import pyvista as pv
from pyvistaqt import QtInteractor
import numpy as np
import trimesh


class ThumbnailIconProvider(QFileIconProvider):
    def __init__(self, cache_dir: str, thumbnail_size: int = 128):
        super().__init__()
        self._cache_dir = cache_dir
        self._thumbnail_size = int(thumbnail_size)
        os.makedirs(self._cache_dir, exist_ok=True)

    def icon(self, fileInfo):  # noqa: N802 (Qt override name)
        try:
            if not fileInfo.isFile():
                return super().icon(fileInfo)

            path = fileInfo.absoluteFilePath()
            ext = os.path.splitext(path)[1].lower()
            if ext not in {".stl", ".3mf"}:
                return super().icon(fileInfo)

            # Avoid heavy work for very large files.
            if fileInfo.size() > 250 * 1024 * 1024:
                return super().icon(fileInfo)

            mtime = int(fileInfo.lastModified().toMSecsSinceEpoch())
            # v2: cache version bump (lets us change render settings without reusing old thumbnails)
            key = hashlib.sha1(f"v2|{path}|{mtime}".encode("utf-8", errors="ignore")).hexdigest()
            thumb_path = os.path.join(self._cache_dir, f"{key}.png")

            if not os.path.exists(thumb_path):
                self._generate_thumbnail(path, thumb_path)

            if os.path.exists(thumb_path):
                return QIcon(thumb_path)
        except Exception:
            pass

        return super().icon(fileInfo)

    def _generate_thumbnail(self, path: str, thumb_path: str) -> None:
        meshes = _load_meshes_for_thumbnail(path)
        if not meshes:
            return

        plotter = pv.Plotter(
            off_screen=True,
            window_size=(self._thumbnail_size, self._thumbnail_size),
        )
        try:
            for mesh in meshes:
                plotter.add_mesh(mesh, color="orange", show_edges=False)
            plotter.view_isometric()
            plotter.reset_camera()
            plotter.screenshot(thumb_path)
        finally:
            plotter.close()


class TwoColumnIconView(QListView):

    def __init__(self, icon_size: int = 256):
        super().__init__()
        self._icon_size = int(icon_size)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setModelColumn(0)
        self.setUniformItemSizes(True)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setWrapping(True)
        self.setWordWrap(True)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setIconSize(QSize(self._icon_size, self._icon_size))
        self._apply_two_column_grid()

    def resizeEvent(self, event):  # noqa: N802 (Qt override name)
        super().resizeEvent(event)
        self._apply_two_column_grid()

    def set_thumbnail_size(self, icon_size: int) -> None:
        size = max(64, int(icon_size))
        size = (size // 16) * 16
        if size == self._icon_size:
            return
        self._icon_size = size
        self.setIconSize(QSize(self._icon_size, self._icon_size))
        self._apply_two_column_grid()

    def _apply_two_column_grid(self) -> None:
        # Roughly 2 columns wide, with some padding for spacing and the label.
        viewport_width = max(1, self.viewport().width())
        cell_width = max(self._icon_size + 20, viewport_width // 2)
        # Allocate enough space for the filename label even on high DPI.
        label_height = self.fontMetrics().height() * 2
        cell_height = self._icon_size + label_height + 40
        self.setGridSize(QSize(cell_width, cell_height))


class PrintFileFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, allowed_exts: set[str]):
        super().__init__()
        self._allowed_exts = {e.lower() for e in allowed_exts}

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:  # noqa: N802
        model = self.sourceModel()
        if model is None:
            return True

        idx = model.index(source_row, 0, source_parent)
        if not idx.isValid():
            return False

        info = model.fileInfo(idx)
        if info.isDir():
            return True

        if not info.isFile():
            return False

        ext = os.path.splitext(info.fileName())[1].lower()
        return ext in self._allowed_exts


def _load_meshes_for_thumbnail(path: str) -> list[pv.PolyData]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        mesh = pv.read(path)
        return [mesh] if mesh is not None else []

    if ext == ".3mf":
        loaded = trimesh.load(path)
        meshes: list[pv.PolyData] = []

        def to_poly(tri_mesh: trimesh.Trimesh) -> pv.PolyData:
            vertices = np.asarray(tri_mesh.vertices, dtype=np.float32)
            faces = np.asarray(tri_mesh.faces, dtype=np.int64)
            faces_pv = np.hstack(
                [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
            ).ravel()
            return pv.PolyData(vertices, faces_pv)

        if isinstance(loaded, trimesh.Scene):
            for node_name in loaded.graph.nodes_geometry:
                geom_name = loaded.graph[node_name][1]
                tri_mesh = loaded.geometry.get(geom_name)
                if tri_mesh is None or tri_mesh.vertices.size == 0 or tri_mesh.faces.size == 0:
                    continue
                transform, _ = loaded.graph.get(node_name)
                tri_mesh = tri_mesh.copy()
                tri_mesh.apply_transform(transform)
                meshes.append(to_poly(tri_mesh))
            return meshes

        if isinstance(loaded, trimesh.Trimesh) and loaded.vertices.size and loaded.faces.size:
            return [to_poly(loaded)]

    return []

class STLViewer(QMainWindow):

    def __init__(self, root_dir: str, *, debug: bool = False):
        super().__init__()
        self._debug = bool(debug)
        self.setWindowTitle("Local 3D Library Viewer")
        self.resize(1000, 600)

        self._log(f"Root dir: {root_dir}")
        self._log(f"Root exists={os.path.exists(root_dir)} isdir={os.path.isdir(root_dir)}")

        # 1. Setup the File System Model (The Logic for finding files)
        self.file_model = QFileSystemModel()
        src_root = self.file_model.setRootPath(root_dir)
        # Keep folders visible even when using name filters.
        self.file_model.setFilter(
            QDir.Filter.AllDirs
            | QDir.Filter.Files
            | QDir.Filter.Drives
            | QDir.Filter.NoDotAndDotDot
        )

        # Filter files via proxy (always keep directories visible).
        self.proxy_model = PrintFileFilterProxyModel({".stl", ".obj", ".3mf"})
        self.proxy_model.setSourceModel(self.file_model)

        cache_root = os.path.join(
            os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
            "PrintStore",
            "thumbs",
        )
        print(f"Thumbnail cache: {cache_root}")
        # Render thumbnails at a higher resolution; the view scales them.
        self.file_model.setIconProvider(ThumbnailIconProvider(cache_root, thumbnail_size=512))

        if self._debug:
            try:
                self.file_model.directoryLoaded.connect(self._on_directory_loaded)
            except Exception:
                pass

        self._log(f"Root index valid={src_root.isValid()}")

        # 2. Setup the Tree View (The List on the left)
        self.tree = QTreeView()
        self.tree.setModel(self.proxy_model)
        self.tree.setRootIndex(self.proxy_model.mapFromSource(src_root))
        self.tree.clicked.connect(self.on_file_click)

        # 2b. Setup the Thumbnail View
        self.thumbs = TwoColumnIconView(icon_size=256)
        self.thumbs.setModel(self.proxy_model)
        self.thumbs.setRootIndex(self.proxy_model.mapFromSource(src_root))
        self.thumbs.clicked.connect(self.on_file_click)

        # Thumbnail navigation UI (Back + Breadcrumbs + Size)
        self._thumb_history: list = []
        self._thumb_nav_bar = QWidget()
        self._thumb_nav_layout = QVBoxLayout(self._thumb_nav_bar)
        self._thumb_nav_layout.setContentsMargins(6, 6, 6, 6)
        self._thumb_nav_layout.setSpacing(4)

        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        self._thumb_back_btn = QToolButton()
        self._thumb_back_btn.setText("←")
        self._thumb_back_btn.setEnabled(False)
        self._thumb_back_btn.clicked.connect(self._thumb_back)
        top_layout.addWidget(self._thumb_back_btn)

        self._thumb_crumb_container = QWidget()
        self._thumb_crumb_layout = QHBoxLayout(self._thumb_crumb_container)
        self._thumb_crumb_layout.setContentsMargins(0, 0, 0, 0)
        self._thumb_crumb_layout.setSpacing(2)
        top_layout.addWidget(self._thumb_crumb_container, 1)

        self._thumb_nav_layout.addWidget(top_row)

        bottom_row = QWidget()
        bottom_layout = QHBoxLayout(bottom_row)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)

        self._thumb_size_label = QLabel("Size")
        bottom_layout.addWidget(self._thumb_size_label)

        self._thumb_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._thumb_size_slider.setRange(96, 512)
        self._thumb_size_slider.setSingleStep(16)
        self._thumb_size_slider.setPageStep(64)
        self._thumb_size_slider.setValue(256)
        self._thumb_size_slider.setFixedWidth(160)
        self._thumb_size_slider.valueChanged.connect(self._on_thumb_size_changed)
        bottom_layout.addWidget(self._thumb_size_slider)
        bottom_layout.addStretch(1)

        self._thumb_nav_layout.addWidget(bottom_row)

        self._thumb_tab = QWidget()
        thumb_tab_layout = QVBoxLayout(self._thumb_tab)
        thumb_tab_layout.setContentsMargins(0, 0, 0, 0)
        thumb_tab_layout.setSpacing(0)
        thumb_tab_layout.addWidget(self._thumb_nav_bar)
        thumb_tab_layout.addWidget(self.thumbs, 1)

        # Tabs to switch between folder list and thumbnails
        self.tabs = QTabWidget()
        self.tabs.addTab(self.tree, "Folders")
        self.tabs.addTab(self._thumb_tab, "Thumbnails")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Keep thumbnail root synced with the currently selected folder in the tree
        self._current_dir_index = src_root
        self._update_thumbnail_nav_ui()
        self.tree.selectionModel().currentChanged.connect(self._on_tree_current_changed)
        
        # 3. Setup the 3D Plotter (The Viewer on the right)
        self.frame = QWidget()
        self.plotter = QtInteractor(self.frame)
        layout = QVBoxLayout()
        layout.addWidget(self.plotter.interactor)
        self.frame.setLayout(layout)

        # 4. Organize Layout (Splitter allows resizing)
        splitter = QSplitter()
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.frame)
        splitter.setSizes([300, 700]) # Initial width ratio
        self.setCentralWidget(splitter)

        # Add a simple coordinate axes
        self.plotter.add_axes()
        self.plotter.reset_camera()
        self.plotter.view_isometric()

    def _log(self, message: str) -> None:
        if not self._debug:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {message}")

    def _on_directory_loaded(self, path: str) -> None:
        try:
            idx = self.file_model.index(path)
            rows = self.file_model.rowCount(idx)
            self._log(f"directoryLoaded: {path} rows={rows}")
        except Exception as e:
            self._log(f"directoryLoaded handler error: {e}")

    def on_file_click(self, index):
        sender = self.sender()
        src_index = self.proxy_model.mapToSource(index)
        file_path = self.file_model.filePath(src_index)

        # If it's a folder, update the thumbnail view to show it.
        if os.path.isdir(file_path):
            # Only record history when navigating from the thumbnail view itself.
            push_history = sender == self.thumbs
            self._set_current_directory_index(src_index, push_history=push_history)
            return

        if os.path.isfile(file_path):
            self.load_mesh(file_path)

    def _set_current_directory_index(self, index, *, push_history: bool = False) -> None:
        if push_history and self._current_dir_index.isValid() and index != self._current_dir_index:
            self._thumb_history.append(self._current_dir_index)
        self._current_dir_index = index
        self.thumbs.setRootIndex(self.proxy_model.mapFromSource(index))
        self._update_thumbnail_nav_ui()

    def _on_tree_current_changed(self, current, previous) -> None:
        try:
            src_current = self.proxy_model.mapToSource(current)
            path = self.file_model.filePath(src_current)
            if os.path.isdir(path):
                self._set_current_directory_index(src_current, push_history=False)
        except Exception:
            pass

    def _on_tab_changed(self, tab_index: int) -> None:
        # Ensure thumbnails always show the last selected directory
        if tab_index == 1:
            try:
                self.thumbs.setRootIndex(self.proxy_model.mapFromSource(self._current_dir_index))
                self._update_thumbnail_nav_ui()
            except Exception:
                pass

    def _thumb_back(self) -> None:
        if not self._thumb_history:
            return
        prev = self._thumb_history.pop()
        if prev.isValid():
            self._set_current_directory_index(prev, push_history=False)

    def _update_thumbnail_nav_ui(self) -> None:
        # Back button state
        self._thumb_back_btn.setEnabled(len(self._thumb_history) > 0)

        # Breadcrumb buttons
        while self._thumb_crumb_layout.count():
            item = self._thumb_crumb_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        try:
            current_path = self.file_model.filePath(self._current_dir_index)
        except Exception:
            current_path = ""

        for label, path in self._split_path_for_breadcrumb(current_path):
            btn = QPushButton(label)
            btn.setFlat(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, p=path: self._jump_to_path(p))
            self._thumb_crumb_layout.addWidget(btn)

    def _on_thumb_size_changed(self, value: int) -> None:
        try:
            self.thumbs.set_thumbnail_size(value)
        except Exception:
            pass

    def _jump_to_path(self, path: str) -> None:
        try:
            idx = self.file_model.index(path)
            if idx.isValid():
                self._set_current_directory_index(idx, push_history=True)
        except Exception:
            pass

    def _split_path_for_breadcrumb(self, path: str) -> list[tuple[str, str]]:
        if not path:
            return [("(none)", "")]

        # Normalize separators for display logic.
        p = path.replace("/", "\\")

        # UNC path: \\server\share\rest...
        if p.startswith("\\\\"):
            parts = [x for x in p.lstrip("\\").split("\\") if x]
            if len(parts) >= 2:
                base = "\\\\" + parts[0] + "\\" + parts[1]
                crumbs: list[tuple[str, str]] = [(base, base)]
                cur = base
                for seg in parts[2:]:
                    cur = cur + "\\" + seg
                    crumbs.append((seg, cur))
                return crumbs

        drive, tail = os.path.splitdrive(p)
        tail_parts = [x for x in tail.strip("\\").split("\\") if x]
        if drive:
            cur = drive + "\\"
            crumbs = [(cur, cur)]
        else:
            cur = "\\"
            crumbs = [("\\", "\\")]

        for seg in tail_parts:
            cur = os.path.join(cur, seg)
            crumbs.append((seg, cur))
        return crumbs

    def load_mesh(self, path):
        try:
            # Clear previous mesh
            self.plotter.clear()
            self.plotter.add_axes()

            # Load and display new mesh
            ext = os.path.splitext(path)[1].lower()
            if ext == ".3mf":
                self._load_3mf_into_plotter(path)
            else:
                mesh = pv.read(path)

            if ext != ".3mf":
                self.plotter.add_mesh(mesh, color="orange", show_edges=True)
            self.plotter.reset_camera()
            self.plotter.view_isometric()
            
        except Exception as e:
            print(f"Error loading file: {e}")

    def _load_3mf_into_plotter(self, path: str) -> None:
        loaded = trimesh.load(path)

        if isinstance(loaded, trimesh.Scene):
            if not loaded.geometry:
                raise ValueError("3MF contains no geometry")

            added_any = False
            for node_name in loaded.graph.nodes_geometry:
                geom_name = loaded.graph[node_name][1]
                tri_mesh = loaded.geometry.get(geom_name)
                if tri_mesh is None:
                    continue
                transform, _ = loaded.graph.get(node_name)
                self._add_trimesh_to_plotter(tri_mesh, transform=transform)
                added_any = True

            if not added_any:
                raise ValueError("3MF scene graph contains no renderable meshes")
            return

        if isinstance(loaded, trimesh.Trimesh):
            self._add_trimesh_to_plotter(loaded, transform=None)
            return

        raise ValueError(f"Unsupported 3MF load result: {type(loaded)}")

    def _add_trimesh_to_plotter(self, tri_mesh: trimesh.Trimesh, transform) -> None:
        if tri_mesh.vertices.size == 0 or tri_mesh.faces.size == 0:
            return

        if transform is not None:
            tri_mesh = tri_mesh.copy()
            tri_mesh.apply_transform(transform)

        poly = self._trimesh_to_pyvista(tri_mesh)

        # Prefer embedded colors when present.
        rgb_cells = self._extract_face_rgb(tri_mesh)
        if rgb_cells is not None and rgb_cells.shape[0] == poly.n_cells:
            # If the entire part is a single solid color, apply it directly.
            if rgb_cells.shape[0] > 0 and np.all(rgb_cells == rgb_cells[0]):
                r, g, b = (int(rgb_cells[0, 0]), int(rgb_cells[0, 1]), int(rgb_cells[0, 2]))
                self.plotter.add_mesh(poly, color=(r / 255.0, g / 255.0, b / 255.0), show_edges=True)
            else:
                poly.cell_data["rgb"] = rgb_cells
                self.plotter.add_mesh(
                    poly,
                    scalars="rgb",
                    rgb=True,
                    show_edges=True,
                    preference="cell",
                )
            return

        rgb_points = self._extract_vertex_rgb(tri_mesh)
        if rgb_points is not None and rgb_points.shape[0] == poly.n_points:
            if rgb_points.shape[0] > 0 and np.all(rgb_points == rgb_points[0]):
                r, g, b = (int(rgb_points[0, 0]), int(rgb_points[0, 1]), int(rgb_points[0, 2]))
                self.plotter.add_mesh(poly, color=(r / 255.0, g / 255.0, b / 255.0), show_edges=True)
            else:
                poly.point_data["rgb"] = rgb_points
                self.plotter.add_mesh(
                    poly,
                    scalars="rgb",
                    rgb=True,
                    show_edges=True,
                    preference="point",
                )
            return

        self.plotter.add_mesh(poly, color="orange", show_edges=True)

    def _trimesh_to_pyvista(self, tri_mesh: trimesh.Trimesh) -> pv.PolyData:
        vertices = np.asarray(tri_mesh.vertices, dtype=np.float32)
        faces = np.asarray(tri_mesh.faces, dtype=np.int64)
        faces_pv = np.hstack(
            [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
        ).ravel()
        return pv.PolyData(vertices, faces_pv)

    def _extract_face_rgb(self, tri_mesh: trimesh.Trimesh):
        try:
            visual = getattr(tri_mesh, "visual", None)
            if visual is None:
                return None
            face_colors = getattr(visual, "face_colors", None)
            if face_colors is None:
                return None
            face_colors = np.asarray(face_colors)
            if face_colors.ndim != 2 or face_colors.shape[0] != tri_mesh.faces.shape[0] or face_colors.shape[1] < 3:
                return None
            rgb = face_colors[:, :3]
            if np.issubdtype(rgb.dtype, np.floating) and rgb.size and rgb.max() <= 1.0:
                rgb = (rgb * 255.0)
            return np.clip(rgb, 0, 255).astype(np.uint8)
        except Exception:
            return None

    def _extract_vertex_rgb(self, tri_mesh: trimesh.Trimesh):
        try:
            visual = getattr(tri_mesh, "visual", None)
            if visual is None:
                return None
            vertex_colors = getattr(visual, "vertex_colors", None)
            if vertex_colors is None:
                return None
            vertex_colors = np.asarray(vertex_colors)
            if vertex_colors.ndim != 2 or vertex_colors.shape[0] != tri_mesh.vertices.shape[0] or vertex_colors.shape[1] < 3:
                return None
            rgb = vertex_colors[:, :3]
            if np.issubdtype(rgb.dtype, np.floating) and rgb.size and rgb.max() <= 1.0:
                rgb = (rgb * 255.0)
            return np.clip(rgb, 0, 255).astype(np.uint8)
        except Exception:
            return None

if __name__ == "__main__":
    app = QApplication(sys.argv)

    args = sys.argv[1:]
    debug = ("--debug" in args) or ("-d" in args) or (os.environ.get("PRINTSTORE_DEBUG") == "1")
    args = [a for a in args if a not in {"--debug", "-d"}]

    # Default to the workspace "files" folder (next to this script)
    default_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")

    # Allow passing a folder on the command line (supports UNC paths like \\server\share)
    arg_folder = args[0] if len(args) > 0 else ""
    arg_folder = os.path.expandvars(os.path.expanduser(arg_folder)) if arg_folder else ""

    if arg_folder and os.path.isdir(arg_folder):
        target_folder = arg_folder
    else:
        # Pick a folder on launch (Cancel falls back to default_folder)
        picked = QFileDialog.getExistingDirectory(
            None,
            "Select 3D Print Folder",
            arg_folder or default_folder,
        )
        target_folder = picked or default_folder

    if not os.path.exists(target_folder) and target_folder == default_folder:
        os.makedirs(target_folder, exist_ok=True)

    window = STLViewer(target_folder, debug=debug)
    window.show()
    sys.exit(app.exec())