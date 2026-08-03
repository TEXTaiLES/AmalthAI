import os
import zipfile


def safe_extract_zip(zip_path, dest_dir):
    """Extract a zip file into dest_dir, rejecting any entry that would land
    outside dest_dir (Zip Slip: '../' path traversal, absolute paths, or
    symlink-like entries pointing outside the target)."""
    dest_dir = os.path.abspath(dest_dir)
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.infolist():
            member_path = os.path.abspath(os.path.join(dest_dir, member.filename))
            if member_path != dest_dir and not member_path.startswith(dest_dir + os.sep):
                raise ValueError(f"Unsafe path in zip entry: {member.filename}")
        z.extractall(dest_dir)