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
        result,reward,is_terminal = Mangala.transition(state, 2)
        print(f"Result: {result}, Reward: {reward}, Is Terminal: {is_terminal}")
        self.assertEqual(result, expected)


    def test_capture_on_empty_own_pit(self):
        state = [0, 0, 0, 0, 1, 0, 0,
                 5, 0, 0, 0, 0, 0, 0]

        expected = [0, 0, 0, 0, 0, 0, 6,
                    0, 0, 0, 0, 0, 0, 0]
        result,reward,is_terminal = Mangala.transition(state, 4)
        print(f"Result: {result}, Reward: {reward}, Is Terminal: {is_terminal}")
        self.assertEqual(result, expected)


    def test_even_capture_opponent_pit(self):
        state = [0, 0, 0, 0, 0, 5, 0,
                 0, 0, 1, 0, 0, 0, 0]

        expected = [0, 0, 0, 0, 0, 1, 3,
                    1, 1, 0, 0, 0, 0, 0]
        result,reward,is_terminal = Mangala.transition(state, 5)
        print(f"Result: {result}, Reward: {reward}, Is Terminal: {is_terminal}")
        self.assertEqual(result, expected)

    def test_skip_opponent_store(self):
        state = [0, 0, 0, 0, 0, 10, 0,
                 0, 0, 0, 0, 0, 0, 5]
        result,reward,is_terminal = Mangala.transition(state[:], 5)
        print(f"Result: {result}, Reward: {reward}, Is Terminal: {is_terminal}")
        self.assertEqual(result[13], 5)

    @patch.object(BaseAgent, 'act', return_value=5)
    def test_game_win(self, mock_act):
        state = [0, 0, 0, 0, 0, 1, 9,
                 0, 0, 1, 1, 0, 0, 11]
        game = Mangala(agent0=BaseAgent("player0"), agent1=BaseAgent("player1"),board=state)
        game.start()

        expected = [0, 0, 0, 0, 0, 0, 12,
                 0, 0, 0, 0, 0, 0, 11]
        result = game.board
        self.assertEqual(result, expected)

    @patch.object(BaseAgent, 'act')
    def test_extra_turn(self, mock_act):
        call_count = {'count': 0}

        def act_side_effect(*args, **kwargs):
            call_count['count'] += 1
            if call_count['count'] == 2:
                raise StopIteration("Ending test after second act call")
            return 3

        mock_act.side_effect = act_side_effect
        state = [0, 0, 0, 4, 0, 1, 9,
                 0, 0, 1, 1, 0, 0, 11]
        game = Mangala(agent0=BaseAgent("player0"), agent1=BaseAgent("player1"), board=state)

        with self.assertRaises(StopIteration):
            game.start()

        self.assertEqual(call_count['count'], 2)
        self.assertEqual(game.player_turn,0)