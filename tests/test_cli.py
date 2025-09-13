import unittest
from unittest.mock import patch, MagicMock
from cli import main

class TestCli(unittest.TestCase):

    @patch('sys.argv', ['cli.py', 'list'])
    @patch('cli.sqlite3.connect')
    def test_list_command(self, mock_connect):
        """Test that the list command calls the db and prints."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        main()
        mock_connect.assert_called_once()
        mock_cursor.execute.assert_called_once_with("SELECT name, github_url FROM repositories")

    @patch('sys.argv', ['cli.py', 'index', 'http://test.com'])
    @patch('cli.index_repository')
    def test_index_command(self, mock_index_repository):
        """Test that the index command calls the index_repository function."""
        main()
        mock_index_repository.assert_called_once_with('http://test.com')

    @patch('sys.argv', ['cli.py', 'chat', 'test-repo', 'question'])
    @patch('cli.chat_with_repo')
    def test_chat_command(self, mock_chat_with_repo):
        """Test that the chat command calls the chat_with_repo function."""
        main()
        mock_chat_with_repo.assert_called_once_with('test-repo', 'question')

if __name__ == '__main__':
    unittest.main()