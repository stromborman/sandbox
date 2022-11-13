import java.util.Arrays;
import java.util.Map;
import java.util.Scanner;

/**
 * A tick-tack-toe game
 */
public class TheGame {

    private static final int SIZE = 3;
    private static int numX = 0;
    private static int numO = 0;
    private static int numE = 0; //E for empty
    private static boolean xWins = false;
    private static boolean oWins = false;
    /**
     * logicBoard an integer matrix representing the game board:
     * X's stored as 1's
     * _'s stored as 0's (ie blanks)
     * O's stored as -1's
     * This structure was chosen so row, col, diag sums could be used to check for a win.
     */
    private static int[][] logicBoard = new int[SIZE][SIZE];
    private static String outcome;

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        fillBoard("_".repeat(9));
        printBoard();
        outcome = "Game not finished";
        int turnNum = 0;

        while (outcome.equals("Game not finished")) {

            turnNum += 1;

            // X goes on odd number turns
            String activePlayer = (turnNum % 2 == 0)? "O" : "X";

            // to hold location of player's move
            int row = 0;
            int col = 0;

            // process player's input and check it is valid
            boolean awaitingValidMove = true;
            while (awaitingValidMove) {

                System.out.println("Player " + activePlayer + " enter your move:");
                String[] inputs = scanner.nextLine().split(" ");
                try {
                    row = Integer.parseInt(inputs[0]) - 1;
                    col = Integer.parseInt(inputs[1]) - 1;
                } catch (NumberFormatException nfe) {
                    System.out.println("You should enter numbers!");
                    System.out.println("Sample input: 1 3 places your mark in the first row and third column.");
                    continue;
                }

                if (row < 0 || row > 2 || col < 0 || col > 2) {
                    System.out.println("Coordinates should be from 1 to " + SIZE +".");
                    System.out.println("Sample input: 1 3 places your mark in the first row and third column.");
                    continue;
                }

                if (logicBoard[row][col] != 0) {
                    System.out.println("This cell is occupied! Choose another one!");
                    continue;
                }
                awaitingValidMove = false;
            }

            // update board at the player's chosen location
            if (activePlayer.equals("X")) {
                logicBoard[row][col] = 1;
                numX += 1;
            } else {
                logicBoard[row][col] = -1;
                numO += 1;
            }
            numE -= 1;

            // print updated board
            printBoard();

            // check for update to winner or draw
            analyzeBoard();
            setOutcome(); // if no winner nor a draw, begin loop again for next player's turn
        }

        // Print the final state of the game and terminate program.
        System.out.println(outcome);
    }

    /**
     * Parses an input of string representing game board and initializes the states: numX, num0, numE, logicBoard
     * @param inputStr string using X,O,_ that represents a tick-tack-toe board (eg "__X_OO_X_")
     */
    private static void fillBoard(String inputStr) {
        Map<Character, Integer> char2int = Map.of(
                'X',1,
                'O',-1,
                '_',0
        );

        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                char entry = inputStr.charAt(SIZE*i + j);
                if (entry == 'X'){
                    numX++;
                } else if (entry == 'O') {
                    numO++;
                } else {
                    numE++;
                }
                logicBoard[i][j] = char2int.get(entry);
            }
        }
    }


    /**
     * Reads from the logicBoard and prints nicely with X's, O's, and blanks.
     */
    private static void printBoard(){
        Map<Integer, Character> int2char = Map.of(
                1,'X',
                -1,'O',
                0,'_'
        );

        System.out.println("-".repeat(9));
        for (int[] row : logicBoard) {
            String rowString = "| ";
            for (int j : row) {
                rowString += int2char.get(j)+" ";
            }
            rowString += "|";
            System.out.println(rowString);
        }
        System.out.println("-".repeat(9));
    }

    /**
     * Interprets the states numX, numO, numE, xWins, oWins into the correct current outcome of game.
     */
    private static void setOutcome() {
        if (Math.abs(numX - numO) > 1) {
            outcome = "Impossible";
        } else if (xWins && oWins) {
            outcome = "Impossible";
        } else if (xWins) {
            outcome = "X wins";
        } else if (oWins) {
            outcome = "O wins";
        } else if (numE > 0) {
            outcome = "Game not finished";
        } else
            outcome = "Draw";
    }

    /**
     * Checks all rows, columns, and diagonals for 3 in a row.  Sets the states xWins and oWins accordingly.
     * This runs in O(SIZE^2) time but for the purposes of running the game it could be rewritten to
     * do its job in an online fashion in O(1) time.  (Store sums in hash table, one new move update
     * the required entries accordingly.)
     */
    private static void analyzeBoard() {
        for (int[] row : logicBoard) {
            int rowSum = Arrays.stream(row).sum();
            parseSum(rowSum);
        }

        for (int col = 0; col < SIZE; col++) {
            int colSum = 0;
            for (int i = 0; i < SIZE; i++) {
                colSum += logicBoard[i][col];
            }
            parseSum(colSum);
        }

        int diagSum = 0;
        int antiDiagSum = 0;
        for (int i = 0; i < SIZE; i++) {
            diagSum += logicBoard[i][i];
            antiDiagSum += logicBoard[SIZE-i-1][i];
        }
        parseSum(diagSum);
        parseSum(antiDiagSum);

    }

    private static void parseSum(int sum) {
        if (sum == 3) {
            xWins = true;
        } else if (sum == -3) {
            oWins = true;
        }
    }
}
