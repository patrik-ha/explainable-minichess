from typing import Tuple
import numpy as np

from minichess.chess.fastchess_utils import B_0, B_1, flat, has_bit, inv_color, set_bit, true_bits, unflat, unset_bit, more_than_one_bit_set, agent_state


class Chess:
    def __init__(
        self,
        bitboards,
        piece_lookup,
        dims,
        diagonal_hash,
        diagonal_magics,
        diagonal_shift,
        straight_hash,
        straight_magics,
        straight_shift,
        PAWN_MOVES_SINGLE,
        PAWN_MOVES_DOUBLE,
        PAWN_ATTACKS,
        KNIGHT_MOVES,
        KING_MOVES,
        DIAGONAL_MOVES,
        STRAIGHT_MOVES,
        CASTLING_EMPTY_MASKS,
        CASTLING_ATTACK_MASKS,
        PROMOTION_MASKS,
        castling_rights=None,
        has_en_passant=None,
        en_passant=None,
        ply_count=0,
        turn=1
    ):
        self.bitboards = bitboards
        self.piece_lookup = piece_lookup
        self.dims = dims

        self.diagonal_magic_shift = diagonal_shift
        self.straight_magic_shift = straight_shift

        self.diagonal_hash_table = diagonal_hash
        self.diagonal_magics = diagonal_magics
        self.straight_hash_table = straight_hash
        self.straight_magics = straight_magics

        self.PAWN_MOVES_SINGLE = PAWN_MOVES_SINGLE
        self.PAWN_MOVES_DOUBLE = PAWN_MOVES_DOUBLE
        self.PAWN_ATTACKS = PAWN_ATTACKS
        self.KNIGHT_MOVES = KNIGHT_MOVES
        self.KING_MOVES = KING_MOVES
        self.DIAGONAL_MOVES = DIAGONAL_MOVES
        self.STRAIGHT_MOVES = STRAIGHT_MOVES
        self.CASTLING_EMPTY_MASKS = CASTLING_EMPTY_MASKS
        self.CASTLING_ATTACK_MASKS = CASTLING_ATTACK_MASKS
        self.castling_rights = castling_rights

        self.PROMOTION_MASKS = PROMOTION_MASKS

        self.has_legal_moves = -1
        self.any_checkers = -1

        if has_en_passant is None:
            self.has_en_passant = False
            self.en_passant = np.array([-1, -1], dtype=np.uint8)
        else:
            self.has_en_passant = has_en_passant
            self.en_passant = en_passant
        self.ply_count_without_adv = ply_count
        self.turn = turn

        self.legal_move_cache = None
        self.promotion_move_cache = None

    def game_result(self):
        """
        Returns the result of the game.

        :return int: -1 if black has won, 1 if white has one, 0 if draw. 
        """
        if self.legal_move_cache is None:
            self.legal_moves()

        if self.ply_count_without_adv > 20 or self.insufficient_material():
            return 0

        if not self.has_legal_moves:
            if self.any_checkers:
                # I lost, mate
                return -1 if self.turn == 1 else 1
            else:
                # Stalemate
                return 0
        return None

    def agent_board_state(self):
        """
        Creates and gives the complete game state as a n x m x d numpy array. Mainly used as input for neural net models.

        :return: A fully specified board state
        :rtype: NDArray[float32]
        """

        return agent_state(self.dims, self.bitboards, self.castling_rights, self.turn, self.en_passant, self.has_en_passant, self.ply_count_without_adv)

    def move_magic(self, occupants: np.uint64, i: np.uint8, j: np.uint8, magic_table: np.ndarray, hash_table: np.ndarray, shift: np.uint8):
        """
        Performs a lookup for a single square in the specified magic bitboard lookup.

        :param uint64 occupants: A bitboard containing all blockers on the board.
        :param uint64 i: Rank of square to consider.
        :param uint8 j: File of square to consider
        :param uint8 magic_table: Lookup table (for the magic bytes themselves) to use.
        :param NDArray[uint64] hash_table: Lookup table (for the available moves given occupants and magic bytes) to use.
        :param uint8 shift: The bit-shift (constant) to use for the given board.
        :return uint64: A bitboard containing the result of the lookup. (available moves from (i, j))
        """
        magic = magic_table[i, j]

        return hash_table[i, j, (occupants * magic) >> np.uint64(64 - shift)]

    def diagonal_move_magic(self, all_pieces: np.uint64, i: np.uint8, j: np.uint8):
        """
        Returns a bitboard designating the amount of available diagonal moves from (i, j), given the current pieces on the board.

        :param uint64 all_pieces: Bitboard containing all the pieces on the board.
        :param uint8 i: Rank of square to consider
        :param uint8 j: File of square to consider
        :return uint64: Bitboard containing all pieces on the board.
        """
        return self.move_magic(all_pieces & self.DIAGONAL_MOVES[i, j], i, j, self.diagonal_magics, self.diagonal_hash_table, self.diagonal_magic_shift) & self.DIAGONAL_MOVES[i, j]

    def straight_move_magic(self, all_pieces: np.uint64, i: np.uint8, j: np.uint8):
        """
        Returns a bitboard designating the amount of available straight-line moves from (i, j), given the current pieces on the board.

        :param uint64 all_pieces: Bitboard containing all the pieces on the board.
        :param uint8 i: Rank of square to consider
        :param uint8 j: File of square to consider
        :return uint64: Bitboard containing all pieces on the board.
        """
        return self.move_magic(all_pieces & self.STRAIGHT_MOVES[i, j], i, j, self.straight_magics, self.straight_hash_table, self.straight_magic_shift) & self.STRAIGHT_MOVES[i, j]

    def move_pieces(self, piece_at: np.int8, origin: Tuple[int, int], target: Tuple[int, int], promotion=-1):
        """
        Perform the operation of moving a piece from a given square to a new square. 
        This means deleting it from its original square, and 'creating' it at the target square. This does not delete enemy pieces at the same square.

        :param int piece_at: Type of piece to move. Given as parameter to skip having to look it up again.
        :param Tuple[int, int] origin: Origin square.
        :param Tuple[int, int] target: Target square.
        :param int promotion: _description_, defaults to -1
        """
        f_from = flat(origin[0], origin[1], self.dims)
        f_to = flat(target[0], target[1], self.dims)

        self.bitboards[self.turn, piece_at] = unset_bit(self.bitboards[self.turn, piece_at], f_from)
        self.piece_lookup[self.turn, origin[0], origin[1]] = -1
        piece_at = piece_at if promotion == -1 else promotion
        self.bitboards[self.turn, piece_at] = set_bit(self.bitboards[self.turn, piece_at], f_to)
        self.piece_lookup[self.turn, target[0], target[1]] = piece_at

    def make_move(self, i: np.uint8, j: np.uint8, dx: np.int8, dy: np.int8, promotion=-1):
        """
        Whole routine for moving a piece from a given square with the given dx, dy deltas.

        :param int8 i: Rank of origin square.
        :param int8 j: File of origin square.
        :param int8 dx: Delta in the rank-dimension (i) to move.
        :param int8 dy: Delta in the file-dimension (j) to move.
        :param int promotion: Piece to promote to, defaults to -1
        """
        
        enemy_turn = inv_color(self.turn)
        piece_at = self.piece_at(i, j, self.turn)
        self.move_pieces(piece_at, (i, j), (i + dx, j + dy), promotion)
        # If this was castling...
        if piece_at == 5 and abs(dy) == 2:
            side = dy > 0
            rook_square_y_from = 0 if side == 0 else self.dims[1] - 1
            rook_square_y_to = (j + dy + 1) if side == 0 else (j + dy - 1)
            self.move_pieces(3, (i, rook_square_y_from), (i, rook_square_y_to))

        # If this voids castling...
        if piece_at == 5:
            self.castling_rights[self.turn] = 0

        if piece_at == 3:
            side = 0 if j == 0 else 1
            back_rank = 0 if self.turn == 0 else self.dims[0] - 1
            if i == back_rank and (j == 0 or j == self.dims[1] - 1):
                self.castling_rights[self.turn, side] = 0

        # A capture (if there's someone already on the square)
        if piece_at == 0 and self.has_en_passant and self.en_passant[0] == i + dx and self.en_passant[1] == j + dy:
            f_to = flat(i, j + dy, self.dims)
        else:
            f_to = flat(i + dx, j + dy, self.dims)

        for piece_type in range(5):
            self.bitboards[enemy_turn, piece_type] = unset_bit(self.bitboards[enemy_turn, piece_type], f_to)

        target = unflat(f_to, self.dims)
        # Update the move counter for forced draw...
        # 'If I just moved a pawn or made a capture'
        if piece_at == 0 or self.piece_at(target[0], target[1], enemy_turn) != -1:
            self.ply_count_without_adv = 0
        else:
            self.ply_count_without_adv += 1
        self.piece_lookup[enemy_turn, target[0], target[1]] = -1

        # If this enables en-passant
        if piece_at == 0 and abs(dx) == 2:
            row_inc = 1 if dx > 0 else -1
            self.has_en_passant = True
            self.en_passant = np.array([i + row_inc, j], dtype=np.int8)
        else:
            self.reset_en_passant()
        self.turn = enemy_turn

        # Empty the cache after making a move
        self.legal_move_cache = None
        self.has_legal_moves = False

    def reset_en_passant(self):
        self.has_en_passant = False
        self.en_passant = np.array([-1, -1], dtype=np.int8)

    def make_null_move(self):
        """Essentially just passes the turn, without making any real move."""
        self.reset_en_passant()
        self.turn = inv_color(self.turn)

    def find_king(self, turn: bool):
        """Returns the position of the king. (i, j)"""
        return self.bit_pos(self.bitboards[turn, 5])

    def find_queen(self, turn: bool):
        """Returns the position of the queen. (i, j)"""
        return self.bit_pos(self.bitboards[turn, 4])

    def get_all_pieces(self, ignore_king: bool, turns=[0, 1]):
        """Gets a bitboard of all the pieces on the board.

        :param _type_ ignore_king: Whether to leave the king out of the bitboard. Useful for calculating occupants.
        :param list turns: Colors/players to include in the bitboard, defaults to [0, 1]
        :return uint64: Bitboard 
        """
        all_pieces = B_0
        for turn in turns:
            for piece_type in range(5 if turn == self.turn and ignore_king else 6):
                all_pieces |= self.bitboards[turn, piece_type]
        return all_pieces

    def insufficient_material(self):
        """Calculates if there is sufficient material to continue the game.

        :return bool: If insufficient material is reached.
        """
        for turn in [0, 1]:
            # I have atleast a pawn, a rook, or a queen -> not insufficient material
            if (self.bitboards[turn, 0] | self.bitboards[turn, 3] | self.bitboards[turn, 4]) != 0:
                return False
            total = 0
            # If each player at most only has a bishop or a knight + king
            for _ in true_bits(self.bitboards[turn, 2] | self.bitboards[turn, 1]):
                total += 1
            if total > 1:
                return False
        return True

    def get_attacked_squares(self, turn: bool, for_king=False):
        """Finds a bitboard of attacked squares by a given player.

        :param bool turn: Player to consider (attacked BY this player).
        :param bool for_king: If the squares are for the king, if so, need to calculate these attacked squares slightly difficult, defaults to False
        :return uint64: Bitboard containing the attacked squares
        """
        attacked = B_0
        # Need to exclude king from blockers if calculating evading checks
        all_pieces = self.get_all_pieces(for_king)
        for bit in true_bits(self.bitboards[turn, 0]):
            i, j = unflat(bit, self.dims)
            attacked |= self.PAWN_ATTACKS[turn, i, j]

        for bit in true_bits(self.bitboards[turn, 1]):
            i, j = unflat(bit, self.dims)
            attacked |= self.KNIGHT_MOVES[i, j]

        for bit in true_bits(self.bitboards[turn, 2] | self.bitboards[turn, 4]):
            i, j = unflat(bit, self.dims)
            attacked |= self.diagonal_move_magic(all_pieces, i, j)

        for bit in true_bits(self.bitboards[turn, 3] | self.bitboards[turn, 4]):
            i, j = unflat(bit, self.dims)
            attacked |= self.straight_move_magic(all_pieces, i, j)

        for bit in true_bits(self.bitboards[turn, 5]):
            i, j = unflat(bit, self.dims)
            attacked |= self.KING_MOVES[i, j]

        return attacked

    def bit_pos(self, board: np.uint64):
        """Finds location (i, j) of first 1 in the bitboard."""
        ind = int(board & -board).bit_length() - 1
        return unflat(ind, self.dims)

    def piece_at(self, i: np.uint8, j: np.uint8, turn: bool):
        """Returns the piece-type of the given color at the given position. -1 if none are found."""
        return self.piece_lookup[turn, i, j]

    def can_castle(self, side: bool, all_pieces: np.uint64, king_danger_squares: np.uint64):
        """Calculates if castling is legal to the current side. If castling is disabled for the given variant, castling rights are pre-set to 0.

        :param bool side: Side to consider
        :param uint64 all_pieces: Bitboard of all pieces on the board
        :param uint64 king_danger_squares: Squares that are threatened for the king
        :return bool: If the king can castle to the given side.
        """
        j = 0 if side == 0 else self.dims[1] - 1
        back_rank = 0 if self.turn == 0 else self.dims[0] - 1
        # Må faktisk ha et tårn der
        if not has_bit(self.bitboards[self.turn, 3], flat(back_rank, j, self.dims)):
            return False
        return self.castling_rights[self.turn, side] and (self.CASTLING_ATTACK_MASKS[self.turn, side] & king_danger_squares) == 0 and (self.CASTLING_EMPTY_MASKS[self.turn, side] & all_pieces) == 0

    def single_to_bitboard(self, i, j):
        """Creates an empty bitboard with the bit at (i, j) set."""
        return set_bit(0, flat(i, j, self.dims))

    def find_checkers(self, all_pieces: np.uint64, enemy_turn: bool, king_pos: Tuple[int, int]):
        """Finds the pieces giving check to the player to move's king.

        :param uint64 all_pieces: Bitboard with all the pieces.
        :param bool enemy_turn: The color of the enemy turn. 
        :param Tuple[int8, int8] king_pos: Position of the king of the player to move.
        :return uint64: Bitboard containing all pieces giving check.
        """
        i, j = king_pos
        checkers = B_0
        # Translates to: knight moves that attack the king, and also has a knight present
        # Meaning: knights that attack our king!
        checkers |= self.KNIGHT_MOVES[i, j] & self.bitboards[enemy_turn, 1]
        # Same, but with diagonally attacking pieces
        checkers |= self.diagonal_move_magic(all_pieces, i, j) & (self.bitboards[enemy_turn, 2] | self.bitboards[enemy_turn, 4])
        # Same, but with pieces attacking in straight lines
        checkers |= self.straight_move_magic(all_pieces, i, j) & (self.bitboards[enemy_turn, 3] | self.bitboards[enemy_turn, 4])

        # Same, but with pawns
        # Have to look at pawn attacks from the king-position in the opposite direction of attack of the enemy pawns
        # Enemies attack "downwards", but king looks upward
        checkers |= self.PAWN_ATTACKS[self.turn, i, j] & self.bitboards[enemy_turn, 0]
        return checkers

    def find_pinned_ray(self, all_pieces: np.uint64, king_pos: Tuple[int, int], attacker: Tuple[int, int], pinned_piece: Tuple[int, int], straight: bool):
        """Finds move-rays of a given piece, given that they might be restricted by an absolute pin.

        :param uint64 all_pieces: Bitboard of all pieces on the board.
        :param Tuple[int, int] king_pos: Position of the king.
        :param Tuple[int, int] attacker: Current attacking piece to consider.
        :param Tuple[int, int] pinned_piece: Piece being pinned to the king.
        :param bool straight: If the pinned rays should be on the straight-lines (if not, on the diagonals)
        :return uint64: Bitboard containing the pin-rays.
        """
        pin_mask = self.single_to_bitboard(pinned_piece[0], pinned_piece[1])
        all_pieces_without_pinned = all_pieces & ~pin_mask
        king_mask = self.bitboards[self.turn, 5]
        attacker_mask = self.single_to_bitboard(attacker[0], attacker[1])
        if straight:
            result = (self.straight_move_magic(all_pieces_without_pinned, king_pos[0], king_pos[1]) | king_mask) & (
                self.straight_move_magic(all_pieces_without_pinned, attacker[0], attacker[1]) | attacker_mask)
        else:
            result = (self.diagonal_move_magic(all_pieces_without_pinned, king_pos[0], king_pos[1]) | king_mask) & (
                self.diagonal_move_magic(all_pieces_without_pinned, attacker[0], attacker[1]) | attacker_mask)
        f = flat(king_pos[0], king_pos[1], self.dims)
        if not has_bit(result, f):
            return np.iinfo(np.uint64).max
        return result

    def is_on_diagonal(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        """Helper, if two points are on same diagonal"""
        return abs(p2[1] - p1[1]) == abs(p2[0] - p1[0])

    def is_on_straight(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        """Helper, if two points are on same straight"""
        return p2[1] == p1[1] or p2[0] == p1[0]

    def find_pinned_pieces(self, all_pieces: np.uint64, enemy_pieces: np.uint64, enemy_turn: bool, king_pos: Tuple[int, int]):
        """Finds a bitboard of all pieces that are absolutely pinned.

        :param uint64 all_pieces: Bitboard of all pieces.
        :param uint64 enemy_pieces: Bitboard of all enemy pieces.
        :param bool enemy_turn: Turn of the pleyr not to move.
        :param Tuple[int, int] king_pos: Position of the king of the player to move.
        :return uint64: Bitboard of the absolutelyl pinned pieces.
        """
        king_straights = self.straight_move_magic(all_pieces, king_pos[0], king_pos[1])
        king_diagonals = self.diagonal_move_magic(all_pieces, king_pos[0], king_pos[1])

        # Find all sliding rays from opponent
        pin_rays = np.full(self.dims, np.iinfo(np.uint64).max, dtype=np.uint64)
        squares_to_check = (self.DIAGONAL_MOVES[king_pos[0], king_pos[1]] | self.STRAIGHT_MOVES[king_pos[0], king_pos[1]]
                            | self.single_to_bitboard(king_pos[0], king_pos[1])) & enemy_pieces
        for bit in true_bits(squares_to_check):
            i, j = unflat(bit, self.dims)
            piece_at = self.piece_at(i, j, enemy_turn)
            found = False
            if self.is_on_straight((i, j), king_pos) and (piece_at == 3 or piece_at == 4):
                straight_pinned = self.straight_move_magic(all_pieces, i, j) & king_straights
                if straight_pinned:
                    # There is a pinned piece here!
                    # There can maximally be one piece per ray.
                    # Now, remove the pinned piece from the board, and calculate hypothetical check-rays
                    pinned_piece = self.bit_pos(straight_pinned)
                    pin_rays[pinned_piece[0], pinned_piece[1]] = self.find_pinned_ray(all_pieces, king_pos, (i, j), pinned_piece, True)
                    found = True
            if self.is_on_diagonal((i, j), king_pos) and not found and (piece_at == 2 or piece_at == 4):
                diagonal_pinned = self.diagonal_move_magic(all_pieces, i, j) & king_diagonals
                if diagonal_pinned:
                    # There is a pinned piece here!
                    # There can maximally be one piece per ray.
                    pinned_piece = self.bit_pos(diagonal_pinned)
                    pin_rays[pinned_piece[0], pinned_piece[1]] = self.find_pinned_ray(all_pieces, king_pos, (i, j), pinned_piece, False)

        return pin_rays

    def find_and_validate_en_passant_moves(self, all_pieces: np.uint64, opp_pawn_row_inc: np.int8, enemy_turn: bool, king_pos: Tuple[int, int]):
        """Finds all valid en-passant moves in the given position.

        :param uint64 all_pieces: Bitboard of all pieces.
        :param int8 opp_pawn_row_inc: dy of the enemy pawns. (The direction of the movement of the enemy pawns.) Either -1 or 1.
        :param bool enemy_turn: Turn of the player that doesn't have the move.
        :param Tuple[int, int] king_pos: Position of the king.
        :return uint64: Bitboard of the valid en-passant moves.
        """
        en_passant_moves = B_0
        if self.has_en_passant:
            # Edge-case: taking en-passant opens up for discovered-check
            # i_act, j_act is where the pawn is also open for attack,
            # i_act + opp_row_inc, j_act is where the pawn actually is.
            i_act, j_act = self.en_passant
            # Two potential en-passant attacks:
            for dx in [-1, 1]:
                to_remove = self.single_to_bitboard(i_act + opp_pawn_row_inc, j_act) | self.single_to_bitboard(i_act + opp_pawn_row_inc, j_act + dx)
                # If there is an intersection between the king and some
                intersections = self.straight_move_magic(all_pieces & ~to_remove, king_pos[0], king_pos[1])
                # Find all sliding rays from opponent
                opponent_straights = B_0
                enemies_to_look_for = (self.bitboards[enemy_turn, 3] | self.bitboards[enemy_turn, 4]) & self.STRAIGHT_MOVES[king_pos[0], king_pos[1]]
                for bit in true_bits(enemies_to_look_for):
                    i, j = unflat(bit, self.dims)
                    opponent_straights |= self.straight_move_magic(all_pieces, i, j)
                # ONLY IF there is no intersection, i.e. the en-passant move doesn't cause check, add it.
                if has_bit(intersections & opponent_straights, flat(king_pos[0], king_pos[1], self.dims)) == 0:
                    en_passant_moves |= self.single_to_bitboard(i_act, j_act)
            # en_passant_moves &= (self.PAWN_ATTACKS[enemy_turn, i_act, j_act] & self.bitboards[self.turn, 0])
        return en_passant_moves

    def legal_moves(self):
        """
        Finds all legal moves. 
        They are given as a matrix of bitboards, where the bitboard at (i, j) designates the legal moves that can be made from that position.
        This is returned along with an a similar matrix designating the available promotions.
        """
        if self.legal_move_cache is not None:
            return self.legal_move_cache, self.promotion_move_cache
        enemy_turn = inv_color(self.turn)
        all_pieces = self.get_all_pieces(False)
        my_pieces = self.get_all_pieces(False, [self.turn])
        enemy_pieces = all_pieces & ~my_pieces
        my_pawn_row_inc = -1 if self.turn == 1 else 1
        opp_pawn_row_inc = my_pawn_row_inc * -1
        # Get squares that are dangerous for the king
        king_danger_squares = self.get_attacked_squares(inv_color(self.turn), True)
        king_pos = self.find_king(self.turn)
        # i, j is the position of our king...
        i, j = king_pos
        king_moves = (self.KING_MOVES[i, j] & ~king_danger_squares) & ~my_pieces

        checkers = self.find_checkers(all_pieces, enemy_turn, king_pos)
        more_than_one_checker = more_than_one_bit_set(checkers)
        if more_than_one_checker:
            # In double-check, only legal thing to do is move the king.
            # This means that all the legal moves consist of moves that move the king.
            legal_moves = np.zeros(self.dims, dtype=np.uint64)
            legal_moves[i, j] = king_moves
            self.has_legal_moves = np.any(legal_moves > 0)
            self.any_checkers = checkers != 0
            self.legal_move_cache = legal_moves
            self.promotion_move_cache = np.zeros_like(legal_moves)
            return legal_moves, np.zeros_like(legal_moves)

        # If we are in check, that limits what we can do.
        # We either have to capture the piece in check, or block it.
        # Need to create two masks for this.

        # If not in check, these masks don't limit anything.
        # Need to have two different masks due to en-passant checks.
        # In that (and only that) case, you can capture a piece by moving to an empty tile.
        # Meaning that you capture it by moving to someplace it isn't, thus requiring two different masks!
        capture_mask = np.iinfo(np.uint64).max
        en_passant_capture_mask = B_0
        push_mask = np.iinfo(np.uint64).max
        # Only a single checker!
        if checkers != 0 and not more_than_one_checker:
            # Can capture the checker...
            capture_mask = checkers

            i, j = self.bit_pos(checkers)
            piece_at = self.piece_at(i, j, enemy_turn)
            # Edge-case: checker is a pawn that was just moved two squares forward, opens up for en-passant
            # Need to add the en-passant square as a possible one to be attacked as well.
            if piece_at == 0 and self.has_en_passant:
                en_passant_capture_mask |= self.single_to_bitboard(self.en_passant[0], self.en_passant[1])
            # If a knight is giving check, it can't be blocked
            if piece_at == 0 or piece_at == 1:
                push_mask = B_0
            else:

                # Find the straight-line moves from the king and from the checker, and see where they intersect.

                # TODO: this now also considers squares "behind" the king, which is not right.
                # Like this: ..R..k..x, in this case, the x would also be a valid intersection, and so would be a valid "block".
                # Not good.

                # ACTUALLY, no it wouldn't! As the king counts as a blocker for the magic :)
                intersections = B_0
                if piece_at == 2:
                    intersections |= self.diagonal_move_magic(all_pieces, i, j) & self.diagonal_move_magic(all_pieces, king_pos[0], king_pos[1])
                if piece_at == 3:
                    intersections |= self.straight_move_magic(all_pieces, i, j) & self.straight_move_magic(all_pieces, king_pos[0], king_pos[1])

                # Stop pinning diagonally and straight at the same time...
                if piece_at == 4:
                    if king_pos[0] == i or king_pos[1] == j:
                        intersections |= self.straight_move_magic(all_pieces, i, j) & self.straight_move_magic(all_pieces, king_pos[0], king_pos[1])
                    else:
                        intersections |= self.diagonal_move_magic(all_pieces, i, j) & self.diagonal_move_magic(all_pieces, king_pos[0], king_pos[1])

                # Remove the actual pieces from the mix, leaving the ray between the checker and the king
                intersections &= (~self.bitboards[self.turn, 5] | ~checkers)
                push_mask = intersections

            # Meaning, any piece that has to move, either has to go to the field indicated by push_mask or capture_mask

        # Pinned pieces
        pin_masks = self.find_pinned_pieces(all_pieces, enemy_pieces, enemy_turn, king_pos)
        # Sweet! Now any piece that falls in this pinned_check_ray has to move only within that.

        en_passant_moves = self.find_and_validate_en_passant_moves(all_pieces, opp_pawn_row_inc, enemy_turn, king_pos)

        legal_moves = np.zeros(self.dims, dtype=np.uint64)
        # Now calculate legal moves from position (i, j)
        promotions = np.zeros(self.dims, dtype=np.uint64)
        # print_bitboard(capture_mask, self.dims)
        # print()
        # print_bitboard(en_passant_moves, self.dims)
        # print()
        for bit in true_bits(my_pieces):
            i, j = unflat(bit, self.dims)
            moves_to_make = B_0
            f = flat(i, j, self.dims)
            piece_at = self.piece_at(i, j, self.turn)
            if piece_at == 0:
                # Can either attack an enemy square, or the en-passant square
                moves_to_make |= self.PAWN_ATTACKS[self.turn, i, j] & (enemy_pieces | en_passant_moves)
                # Pushing pawns one step, if there are no pieces in the way
                single_pawn_moves = (self.PAWN_MOVES_SINGLE[self.turn, i, j] & ~all_pieces)
                moves_to_make |= single_pawn_moves
                # If a pawn can't be pushed one square forward, it is blocked from pushing two aswell
                if single_pawn_moves != 0:
                    moves_to_make |= (self.PAWN_MOVES_DOUBLE[self.turn, i, j] & ~all_pieces)

                promotions[i, j] |= (moves_to_make & self.PROMOTION_MASKS[self.turn])

            if piece_at == 1:
                moves_to_make |= (self.KNIGHT_MOVES[i, j] & ~my_pieces)
            if piece_at == 2 or piece_at == 4:
                moves_to_make |= (self.diagonal_move_magic(all_pieces, i, j) & ~my_pieces)
            if piece_at == 3 or piece_at == 4:
                moves_to_make |= (self.straight_move_magic(all_pieces, i, j) & ~my_pieces)
            if piece_at == 5:
                moves_to_make |= king_moves
                # Castle left
                if self.can_castle(0, all_pieces, king_danger_squares):
                    moves_to_make |= self.single_to_bitboard(i, j - 2)
                if self.can_castle(1, all_pieces, king_danger_squares):
                    moves_to_make |= self.single_to_bitboard(i, j + 2)
            # Remove moves that don't deal with check if necessary
            if piece_at != 5 and piece_at != 0:
                moves_to_make &= (capture_mask | push_mask)
            if piece_at == 0:
                moves_to_make &= (capture_mask | en_passant_capture_mask | push_mask)
            # If it's pinned, restrict the moves to the pinned ray
            moves_to_make &= pin_masks[i, j]
            legal_moves[i, j] = moves_to_make

        # Now legal_moves is a (m x n) matrix with bitboards designating the legal moves from the field (i, j)
        # And promotions is a (m x n) matrix with bitboards designating that any (pawn)moves to the given square is a promotion
        self.has_legal_moves = np.any(legal_moves > 0)
        self.any_checkers = checkers != 0
        self.legal_move_cache = legal_moves
        self.promotion_move_cache = promotions
        return legal_moves, promotions

    def copy(self):
        return Chess(
            self.bitboards.copy(),
            self.piece_lookup.copy(),
            self.dims, self.diagonal_hash_table, self.diagonal_magics, self.diagonal_magic_shift, self.straight_hash_table, self.straight_magics, self.straight_magic_shift,
            self.PAWN_MOVES_SINGLE,
            self.PAWN_MOVES_DOUBLE,
            self.PAWN_ATTACKS,
            self.KNIGHT_MOVES,
            self.KING_MOVES,
            self.DIAGONAL_MOVES,
            self.STRAIGHT_MOVES,
            self.CASTLING_EMPTY_MASKS,
            self.CASTLING_ATTACK_MASKS,
            self.PROMOTION_MASKS,
            self.castling_rights.copy(),
            self.has_en_passant, self.en_passant.copy(),
            self.ply_count_without_adv, self.turn)
