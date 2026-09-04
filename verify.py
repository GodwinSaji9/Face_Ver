import json
import os
import sys
from typing import Optional
from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

# Force UTF-8 encoding on standard streams
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from src.blockchain_engine import BlockchainEngine, CONTRACT_CACHE_FILE

app = typer.Typer(
    name="face-blockchain-verifier",
    help="Standalone Blockchain Record Verifier for Face ID Ledger",
    add_completion=False,
)
console = Console()


@app.command()
def check(
    record_id: int = typer.Option(
        0,
        "--record-id",
        "-i",
        help="Record ID index in FaceMatchRegistry contract to inspect and verify.",
    ),
    contract_address: Optional[str] = typer.Option(
        None,
        "--contract",
        "-c",
        help="FaceMatchRegistry contract address (defaults to .last_contract or CONTRACT_ADDRESS).",
    ),
    rpc_url: str = typer.Option(
        "http://127.0.0.1:8545",
        "--rpc-url",
        "-r",
        help="Ethereum JSON-RPC URL (Foundry Anvil default: http://127.0.0.1:8545).",
    ),
    expected_face_hash: Optional[str] = typer.Option(
        None,
        "--expected-face-hash",
        "-f",
        help="Expected SHA-256 face hash to verify against the on-chain record.",
    ),
    expected_match_hash: Optional[str] = typer.Option(
        None,
        "--expected-match-hash",
        "-m",
        help="Expected SHA-256 match hash to verify against the on-chain record.",
    ),
    tamper_test: bool = typer.Option(
        False,
        "--tamper-test",
        help="Execute an intentional tamper test to demonstrate the smart contract catches altered data.",
    ),
):
    """
    Independently inspects an on-chain face match record and validates its cryptographic integrity.
    """
    console.print(
        Panel.fit(
            "[bold cyan]🔍 ON-CHAIN FACE RECORD VERIFIER[/bold cyan]\n"
            "[dim]Querying Foundry Anvil Immutable Ledger for Record Integrity[/dim]",
            border_style="bright_blue",
        )
    )

    blockchain = BlockchainEngine(
        rpc_url=rpc_url,
        contract_address=contract_address,
    )
    blockchain.ensure_connected(auto_start=True)

    if not blockchain.contract_address:
        console.print("[bold red]❌ Error: No contract address specified and .last_contract not found.[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Connecting to Anvil at[/cyan] [bold]{rpc_url}[/bold]")
    console.print(f"[cyan]Querying FaceMatchRegistry contract at[/cyan] [bold yellow]{blockchain.contract_address}[/bold yellow]")
    console.print(f"[cyan]Fetching Record ID:[/cyan] [bold white]#{record_id}[/bold white]\n")

    # Fetch raw record from contract
    contract = blockchain.get_contract()
    total_records = contract.functions.recordCount().call()

    if record_id >= total_records:
        console.print(
            f"[bold red]❌ Record #{record_id} does not exist! "
            f"Total records in registry: {total_records}[/bold red]"
        )
        raise typer.Exit(code=1)

    raw = contract.functions.getRecord(record_id).call()
    on_chain_face_hex = raw[0].hex()
    on_chain_match_hex = raw[1].hex()
    match_url = raw[2]
    platform = raw[3]
    metadata_raw = raw[4]
    block_ts = raw[5]
    recorder = raw[6]

    try:
        metadata_dict = json.loads(metadata_raw)
    except Exception:
        metadata_dict = {"raw": metadata_raw}

    # Display On-Chain Data
    table = Table(title=f"On-Chain Record #{record_id} Details", border_style="dim", header_style="bold magenta")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Record ID", f"#{record_id}")
    table.add_row("Contract Address", blockchain.contract_address)
    table.add_row("Face Hash (on-chain)", f"[bold white]{on_chain_face_hex}[/bold white]")
    table.add_row("Match Hash (on-chain)", f"[bold white]{on_chain_match_hex}[/bold white]")
    table.add_row("Discovered Match URL", f"[underline blue]{match_url}[/underline blue]")
    table.add_row("Platform", f"[bold yellow]{platform}[/bold yellow]")
    table.add_row("Block Timestamp", str(block_ts))
    table.add_row("Recorded By", recorder)
    if "model" in metadata_dict:
        table.add_row("Model / Detector", f"{metadata_dict.get('model')} / {metadata_dict.get('detector')}")
    if "confidence" in metadata_dict:
        table.add_row("Detection Confidence", f"{metadata_dict.get('confidence'):.4f}")
    if "title" in metadata_dict:
        table.add_row("Post Title", metadata_dict.get("title"))
    console.print(table)

    # Verification checks
    target_face = expected_face_hash or on_chain_face_hex
    target_match = expected_match_hash or on_chain_match_hex

    verification = blockchain.verify_record(
        record_id=record_id,
        expected_face_hash=target_face,
        expected_match_hash=target_match,
    )

    console.print("\n[bold]Cryptographic Verification:[/bold]")
    if verification.is_verified:
        console.print(
            Panel(
                "[bold green]✅ RECORD VERIFIED: ON-CHAIN DATA IS AUTHENTIC & UNTAMPERED[/bold green]\n"
                f"• Target Face Hash: {target_face}\n"
                f"• Target Match Hash: {target_match}\n"
                f"• On-chain Face Hash: {on_chain_face_hex}\n"
                f"• On-chain Match Hash: {on_chain_match_hex}\n"
                f"• Verification via Smart Contract: PASSED",
                title="Integrity Status",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                "[bold red]❌ RECORD FAILED VERIFICATION: HASH MISMATCH DETECTED![/bold red]",
                title="Integrity Status",
                border_style="red",
            )
        )

    # Optional Tamper Demonstration
    if tamper_test:
        console.print("\n[bold yellow]═══ RUNNING SIMULATED TAMPER DEMONSTRATION ═══[/bold yellow]")
        fake_face = "0" * 64
        tamper_result = blockchain.verify_record(
            record_id=record_id,
            expected_face_hash=fake_face,
            expected_match_hash=target_match,
        )
        console.print(f"Testing altered face hash: [bold red]{fake_face}[/bold red]")
        console.print(f"Smart contract verifyRecord() returned: [bold]{tamper_result.is_verified}[/bold]")
        if not tamper_result.is_verified:
            console.print("[bold green]✅ Tamper defense verified! The blockchain contract successfully rejected the altered face hash.[/bold green]")


if __name__ == "__main__":
    app()
