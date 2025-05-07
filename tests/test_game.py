import unittest
from unittest.mock import patch

from agents.base_agent import BaseAgent
from mangala.mangala import Mangala


class MangalaTest(unittest.TestCase):

    def test_normal_distribution(self):
        state = [4, 4, 4, 4, 4, 4, 0,
                 4, 4, 4, 4, 4, 4, 0]

        expected = [4, 4, 1, 5, 5, 5, 0,
                    4, 4, 4, 4, 4, 4, 0]
        result = Mangala.transition(state, 2)
        self.assertEqual(result, expected)


    def test_capture_on_empty_own_pit(self):
        state = [0, 0, 0, 0, 1, 0, 0,
                 5, 0, 0, 0, 0, 0, 0]

        expected = [0, 0, 0, 0, 0, 0, 6,
                    0, 0, 0, 0, 0, 0, 0]
        result = Mangala.transition(state, 4)
        self.assertEqual(result, expected)


    def test_even_capture_opponent_pit(self):
        state = [0, 0, 0, 0, 0, 5, 0,
                 0, 0, 1, 0, 0, 0, 0]

        expected = [0, 0, 0, 0, 0, 1, 3,
                    1, 1, 0, 0, 0, 0, 0]
        result = Mangala.transition(state, 5)
        self.assertEqual(result, expected)

    def test_skip_opponent_store(self):
        state = [0, 0, 0, 0, 0, 10, 0,
                 0, 0, 0, 0, 0, 0, 5]
        result = Mangala.transition(state[:], 5)
        self.assertEqual(result[13], 5)

    @patch.object(BaseAgent, 'act', return_value=5)
    def test_agent_action_patched(self, mock_act):
        state = [0, 0, 0, 0, 0, 1, 9,
                 0, 0, 1, 1, 0, 0, 11]
        game = Mangala(agent0=BaseAgent("player0"), agent1=BaseAgent("player1"),board=state)
        game.start()

        expected = [0, 0, 0, 0, 0, 0, 12,
                 0, 0, 0, 0, 0, 0, 11]
        result = game.board
        self.assertEqual(result, expected)