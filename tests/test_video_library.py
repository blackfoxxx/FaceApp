import unittest
import json
import os
import shutil
import time
import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['numpy'].__version__ = '1.20.0'
sys.modules['numpy'].linalg.norm.return_value = 1.0
sys.modules['insightface'] = MagicMock()
sys.modules['insightface.app'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.metrics.pairwise'] = MagicMock()
sys.modules['imagehash'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['torch'] = MagicMock()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, VIDEO_FACES_DB_FILE

class VideoLibraryTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()
        self.video_faces_dir = os.path.join(app.static_folder, 'video_faces')
        if not os.path.exists(self.video_faces_dir):
            os.makedirs(self.video_faces_dir)
            
        # Create a dummy face file
        self.dummy_face_id = "test_face.jpg"
        self.dummy_face_path = os.path.join(self.video_faces_dir, self.dummy_face_id)
        with open(self.dummy_face_path, 'w') as f:
            f.write("dummy content")
            
        # Backup DB if exists
        self.db_backup = None
        if os.path.exists(VIDEO_FACES_DB_FILE):
            shutil.copy2(VIDEO_FACES_DB_FILE, VIDEO_FACES_DB_FILE + '.bak')
            
        # Clear DB for test
        if os.path.exists(VIDEO_FACES_DB_FILE):
            os.remove(VIDEO_FACES_DB_FILE)

    def tearDown(self):
        # Restore DB
        if os.path.exists(VIDEO_FACES_DB_FILE + '.bak'):
            shutil.move(VIDEO_FACES_DB_FILE + '.bak', VIDEO_FACES_DB_FILE)
            
        # Cleanup dummy files
        if os.path.exists(self.dummy_face_path):
            os.remove(self.dummy_face_path)
            
        # Cleanup created video dir
        video_dir = os.path.join(self.video_faces_dir, 'test_video.mp4')
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
            
        video_dir2 = os.path.join(self.video_faces_dir, 'another_video.mp4')
        if os.path.exists(video_dir2):
            shutil.rmtree(video_dir2)

    def test_save_and_retrieve_library(self):
        # Test Save
        payload = {
            'video_name': 'test_video.mp4',
            'faces': [
                {
                    'id': self.dummy_face_id,
                    'timestamp': 10.5,
                    'timestamp_str': '00:00:10',
                    'score': 0.95
                }
            ]
        }
        
        rv = self.app.post('/save_video_faces', 
                           data=json.dumps(payload),
                           content_type='application/json')
        data = json.loads(rv.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['saved_count'], 1)
        self.assertEqual(data['video_group'], 'test_video.mp4')
        
        # Verify file copied
        saved_path = os.path.join(self.video_faces_dir, 'test_video.mp4', self.dummy_face_id)
        self.assertTrue(os.path.exists(saved_path))
        
        # Test Save Another Video (for sorting)
        payload2 = {
            'video_name': 'another_video.mp4',
            'faces': [
                {
                    'id': self.dummy_face_id,
                    'timestamp': 5.0,
                    'timestamp_str': '00:00:05',
                    'score': 0.90
                }
            ]
        }
        self.app.post('/save_video_faces', 
                      data=json.dumps(payload2),
                      content_type='application/json')
        
        # Test Retrieve (Sorting)
        rv = self.app.get('/get_video_library')
        data = json.loads(rv.data)
        self.assertTrue(data['success'])
        library = data['library']
        self.assertEqual(len(library), 2)
        # 'another_video.mp4' comes before 'test_video.mp4'
        self.assertEqual(library[0]['video_name'], 'another_video.mp4')
        self.assertEqual(library[1]['video_name'], 'test_video.mp4')
        
        # Test Delete Item (Face)
        face_id_to_delete = library[1]['faces'][0]['id'] # test_video.mp4/test_face.jpg
        rv = self.app.post('/delete_video_library_item',
                           data=json.dumps({'video_name': 'test_video.mp4', 'face_id': face_id_to_delete}),
                           content_type='application/json')
        data = json.loads(rv.data)
        self.assertTrue(data['success'])
        
        # Verify face deleted from DB
        rv = self.app.get('/get_video_library')
        library = json.loads(rv.data)['library']
        # Find test_video group
        test_video_group = next(g for g in library if g['video_name'] == 'test_video.mp4')
        self.assertEqual(len(test_video_group['faces']), 0)
        
        # Test Delete Group
        rv = self.app.post('/delete_video_library_item',
                           data=json.dumps({'video_name': 'another_video.mp4'}),
                           content_type='application/json')
        
        rv = self.app.get('/get_video_library')
        library = json.loads(rv.data)['library']
        # another_video should be gone
        self.assertFalse(any(g['video_name'] == 'another_video.mp4' for g in library))

if __name__ == '__main__':
    unittest.main()
