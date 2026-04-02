"""
Test Harness for VLSI Cell Placement Challenge
==============================================

This script runs the placement optimizer on 10 randomly generated netlists
of various sizes and reports metrics for leaderboard submission.

Usage:
    python test_placement.py

Metrics Reported:
    - Average Overlap: (num cells with overlaps / total num cells)
    - Average Wirelength: (total wirelength / num nets) / sqrt(total area)
      This normalization allows fair comparison across different design sizes.

Note: This test uses the default hyperparameters from train_placement() in
vb_playground.py. The challenge is to implement the overlap loss function,
not to tune hyperparameters.
"""

import argparse
import os
import time

import torch

# Import from the challenge file
from placement import (
    calculate_normalized_metrics,
    generate_placement_input,
    train_placement,
)


# Test case configurations: (test_id, num_macros, num_std_cells, seed)
TEST_CASES = [
    # Small designs
    (1, 2, 20, 1001),
    (2, 3, 25, 1002),
    (3, 2, 30, 1003),
    # Medium designs
    (4, 3, 50, 1004),
    (5, 4, 75, 1005),
    (6, 5, 100, 1006),
    # Large designs
    (7, 5, 150, 1007),
    (8, 7, 150, 1008),
    (9, 8, 200, 1009),
    (10, 10, 2000, 1010),
    # Realistic designs
    (11, 10, 10000, 1011),
    (12, 10, 100000, 1012),
]


def run_placement_test(
    test_id,
    num_macros,
    num_std_cells,
    seed=None,
    verbose=False,
):
    """Run placement optimization on a single test case.

    All cells start at the origin (0, 0). Phase 1 drives them to their
    connectivity centroid; phase 2 eliminates overlaps; phase 3 fine-tunes
    wirelength while guarding against re-overlap.

    Args:
        test_id: Test case identifier
        num_macros: Number of macro cells
        num_std_cells: Number of standard cells
        seed: Random seed for reproducibility

    Returns:
        Dictionary with test results and metrics
    """
    if seed:
        torch.manual_seed(seed)

    # Generate netlist (cells initialised at origin by default)
    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros, num_std_cells
    )

    # Run optimization with default hyperparameters
    start_time = time.time()
    result = train_placement(
        cell_features,
        pin_features,
        edge_list,
        verbose=verbose,
    )
    elapsed_time = time.time() - start_time

    # Calculate final metrics using shared implementation
    final_cell_features = result["final_cell_features"]
    metrics = calculate_normalized_metrics(final_cell_features, pin_features, edge_list)

    return {
        "test_id": test_id,
        "num_macros": num_macros,
        "num_std_cells": num_std_cells,
        "total_cells": metrics["total_cells"],
        "num_nets": metrics["num_nets"],
        "seed": seed,
        "elapsed_time": elapsed_time,
        # Final metrics
        "num_cells_with_overlaps": metrics["num_cells_with_overlaps"],
        "overlap_ratio": metrics["overlap_ratio"],
        "normalized_wl": metrics["normalized_wl"],
        # Raw data for optional visualization
        "_initial_cell_features": result["initial_cell_features"],
        "_phase1_cell_features": result["phase1_cell_features"],
        "_phase2_cell_features": result["phase2_cell_features"],
        "_final_cell_features": result["final_cell_features"],
        "_pin_features": pin_features,
        "_edge_list": edge_list,
        "_loss_history": result["loss_history"],
    }


def run_all_tests(visualize=False, output_dir="output", verbose=False, extra_credit=False):
    """Run all test cases and compute aggregate metrics.

    Args:
        visualize: If True, save placement and loss plots for each test case.
        output_dir: Directory to write visualization images into.
        extra_credit: If True, also run tests 11 and 12.

    Returns:
        Dictionary with all test results and aggregate statistics
    """
    if visualize:
        from visualize import plot_placement, plot_loss_history
        os.makedirs(output_dir, exist_ok=True)

    # Tests 1–10 are scored; 11–12 are extra credit
    cases = TEST_CASES if extra_credit else TEST_CASES[:10]

    print("=" * 70)
    print("PLACEMENT CHALLENGE TEST SUITE")
    print("=" * 70)
    print(f"\nRunning {len(cases)} test cases with various netlist sizes...")
    print("Using default hyperparameters from train_placement()")
    print()

    all_results = []

    for idx, (test_id, num_macros, num_std_cells, seed) in enumerate(cases, 1):
        size_category = (
            "Small" if num_std_cells <= 30
            else "Medium" if num_std_cells <= 100
            else "Large"
        )

        print(f"Test {idx}/{len(TEST_CASES)}: {size_category} ({num_macros} macros, {num_std_cells} std cells)")
        print(f"  Seed: {seed}")

        # Run test
        result = run_placement_test(
            test_id,
            num_macros,
            num_std_cells,
            seed,
            verbose=verbose,
        )

        all_results.append(result)

        # Print summary
        status = "✓ PASS" if result["num_cells_with_overlaps"] == 0 else "✗ FAIL"
        print(f"  Overlap Ratio: {result['overlap_ratio']:.4f} ({result['num_cells_with_overlaps']}/{result['total_cells']} cells)")
        print(f"  Normalized WL: {result['normalized_wl']:.4f}")
        print(f"  Time: {result['elapsed_time']:.2f}s")
        print(f"  Status: {status}")

        if visualize:
            prefix = os.path.join(output_dir, f"test_{test_id:02d}")
            plot_placement(
                result["_initial_cell_features"],
                result["_final_cell_features"],
                result["_pin_features"],
                result["_edge_list"],
                phase1_cell_features=result["_phase1_cell_features"],
                phase2_cell_features=result["_phase2_cell_features"],
                filename=f"{prefix}_placement.png",
            )
            plot_loss_history(
                result["_loss_history"],
                filename=f"{prefix}_loss.png",
            )

        print()

    # Compute aggregate statistics over tests 1–10 only
    scored = [r for r in all_results if r["test_id"] <= 10]
    avg_overlap_ratio = sum(r["overlap_ratio"] for r in scored) / len(scored)
    avg_normalized_wl = sum(r["normalized_wl"] for r in scored) / len(scored)
    total_time = sum(r["elapsed_time"] for r in all_results)

    # Print aggregate results
    print("=" * 70)
    print("FINAL RESULTS  (tests 1–10)")
    print("=" * 70)
    print(f"Average Overlap:     {avg_overlap_ratio:.4f}")
    print(f"Average Wirelength:  {avg_normalized_wl:.4f}")
    print(f"Total Runtime:       {total_time:.2f}s")
    print()

    # Per-test breakdown table
    print("Per-test breakdown:")
    print(f"  {'Test':>4}  {'Cells':>6}  {'Overlap':>9}  {'Norm WL':>9}  {'Time(s)':>8}  Status")
    print(f"  {'-'*4}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*8}  ------")
    for r in all_results:
        status = "PASS" if r["num_cells_with_overlaps"] == 0 else "FAIL"
        ec = " *" if r["test_id"] > 10 else ""
        print(f"  {r['test_id']:>4}  {r['total_cells']:>6}  {r['overlap_ratio']:>9.4f}  "
              f"{r['normalized_wl']:>9.4f}  {r['elapsed_time']:>8.2f}  {status}{ec}")
    print()

    # Leaderboard submission row
    print("=" * 70)
    print("LEADERBOARD SUBMISSION")
    print("=" * 70)
    print("Copy the row below into README.md (fill in your name and notes):\n")
    print(f"| ??   | <Your Name>     | {avg_overlap_ratio:.4f}      "
          f"| {avg_normalized_wl:.4f}          | {total_time:.2f}        |                      |")
    print()

    return {
        "avg_overlap": avg_overlap_ratio,
        "avg_wirelength": avg_normalized_wl,
        "total_time": total_time,
    }


def main():
    """Main entry point for the test suite."""
    parser = argparse.ArgumentParser(description="VLSI placement challenge test suite")
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate placement and loss plots for each test case",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory to write visualization images into (default: output/)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-epoch training progress for each test case",
    )
    parser.add_argument(
        "--extra-credit", action="store_true",
        help="Also run tests 11 and 12 (10K and 100K cells)",
    )
    args = parser.parse_args()

    run_all_tests(
        visualize=args.visualize,
        output_dir=args.output_dir,
        verbose=args.verbose,
        extra_credit=args.extra_credit,
    )


if __name__ == "__main__":
    main()
