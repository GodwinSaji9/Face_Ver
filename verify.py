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
from src.face_engine import FaceEngine
from src.utils import select_image_file

app = typer.Typer(
    name="face-blockchain-verifier",
    help="Standalone Blockchain Record Verifier for Face ID & Multi-Site Consensus Ledger",
    add_completion=False,
)
console = Console()


@app.command()
def check(
    record_id: int = typer.Option(
        0,
        "--record-id",
        "-r",
        help="Record ID or Consensus ID in FaceMatchRegistry contract to inspect and verify.",
    ),
    is_consensus: bool = typer.Option(
        True,
        "--consensus/--single",
        help="Whether to verify a multi-site consensus record (default) or single-site record.",
    ),
    image: Optional[str] = typer.Option(
        None,
        "--image",
        "-i",
        help="Path to photo to re-extract face embedding and verify against on-chain record.",
    ),
    choose_file: bool = typer.Option(
        False,
        "--choose-file",
        "--browse",
        "-b",
        help="Open native system file dialog to choose an image from your computer to verify.",
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
        help="Ethereum JSON-RPC URL (Foundry Anvil default: http://127.0.0.1:8545).",
    ),
    expected_face_hash: Optional[str] = typer.Option(
        None,
        "--expected-face-hash",
        "-f",
        help="Expected SHA-256 face hash to verify against the on-chain record.",
    ),
    expected_merkle_root: Optional[str] = typer.Option(
        None,
        "--expected-merkle-root",
        "-m",
        help="Expected Merkle root to verify against the on-chain consensus record.",
    ),
    tamper_test: bool = typer.Option(
        False,
        "--tamper-test",
        help="Execute an intentional tamper test to demonstrate the smart contract catches altered data.",
    ),
):
    """
    Independently inspects an on-chain face match record or multi-site consensus record
    and validates its cryptographic integrity on Foundry Anvil.
    """
    console.print(
        Panel.fit(
            "[bold cyan]🔍 ON-CHAIN CONSENSUS & IDENTITY RECORD VERIFIER[/bold cyan]\n"
            "[dim]Querying Foundry Anvil Immutable Ledger for Multi-Site Record Integrity[/dim]",
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
    console.print(f"[cyan]Target Record ID:[/cyan] [bold white]#{record_id}[/bold white] | [cyan]Type:[/cyan] {'Multi-Site Consensus' if is_consensus else 'Single Record'}\n")

    # Resolve image if user requested file browsing or passed image path
    if choose_file:
        console.print("[cyan]📂 Opening system file dialog to choose an image to verify...[/cyan]")
        chosen = select_image_file(prompt_if_cancelled=True)
        if chosen:
            image = chosen

    if image:
        if not os.path.exists(image):
            console.print(f"[bold red]❌ Specified image not found:[/bold red] {image}")
            raise typer.Exit(code=1)
        console.print(f"[cyan]Extracting face biometric fingerprint from:[/cyan] [bold white]{image}[/bold white]")
        engine = FaceEngine()
        face_res = engine.analyze(image)
        expected_face_hash = face_res.face_hash
        console.print(f"[green]✔ Extracted face hash:[/green] [bold yellow]{expected_face_hash}[/bold yellow]\n")

    contract = blockchain.get_contract()

    if is_consensus:
        # Multi-Site Consensus Record Verification
        total_consensus = contract.functions.consensusCount().call()
        if record_id >= total_consensus:
            # Fallback to single record if no consensus records exist yet
            if contract.functions.recordCount().call() > 0:
                is_consensus = False
            else:
                console.print(f"[bold red]❌ Consensus Record #{record_id} does not exist! Total in registry: {total_consensus}[/bold red]")
                raise typer.Exit(code=1)

    if is_consensus:
        raw_rec = contract.functions.getConsensusRecord(record_id).call()
        on_chain_face_hex = raw_rec[0].hex()
        on_chain_merkle_hex = raw_rec[1].hex()
        entity_name = raw_rec[2]
        platforms = list(raw_rec[3])
        match_urls = list(raw_rec[4])
        score_scaled = raw_rec[5]
        verified_count = raw_rec[6]
        meta_raw = raw_rec[7]
        block_ts = raw_rec[8]
        recorder = raw_rec[9]

        table = Table(title=f"On-Chain Multi-Site Consensus Record #{record_id}", border_style="dim", header_style="bold magenta")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Consensus Record ID", f"#{record_id}")
        table.add_row("Contract Address", blockchain.contract_address)
        table.add_row("Finalized Entity Name", f"[bold yellow]{entity_name}[/bold yellow]")
        table.add_row("Consensus Score", f"{score_scaled / 100.0:.2f}%")
        table.add_row("Verified Sites Count", f"{verified_count} platforms")
        table.add_row("Verified Platforms", ", ".join(platforms))
        table.add_row("Face Hash (on-chain)", on_chain_face_hex)
        table.add_row("Merkle Root (on-chain)", on_chain_merkle_hex)
        table.add_row("Block Timestamp", str(block_ts))
        table.add_row("Recorded By", recorder)
        console.print(table)

        # Verification check
        target_face = expected_face_hash or on_chain_face_hex
        target_merkle = expected_merkle_root or on_chain_merkle_hex

        verification = blockchain.verify_consensus_record(
            consensus_id=record_id,
            expected_face_hash=target_face,
            expected_merkle_root=target_merkle,
        )

        console.print("\n[bold]Cryptographic Verification:[/bold]")
        if verification.is_verified:
            console.print(
                Panel(
                    f"[bold green]✅ CONSENSUS RECORD VERIFIED: MULTI-SITE DATA IS AUTHENTIC & UNTAMPERED[/bold green]\n"
                    f"• Finalized Entity: [bold white]{verification.entity_name}[/bold white]\n"
                    f"• Target Face Hash: {target_face}\n"
                    f"• Target Merkle Root: {target_merkle}\n"
                    f"• On-chain Face Hash: {on_chain_face_hex}\n"
                    f"• On-chain Merkle Root: {on_chain_merkle_hex}\n"
                    f"• Smart Contract verifyConsensusRecord(): PASSED",
                    title="Consensus Integrity Status",
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

        # Tamper defense test
        if tamper_test:
            console.print("\n[bold yellow]═══ RUNNING SIMULATED TAMPER DEMONSTRATION ═══[/bold yellow]")
            fake_face = "0" * 64
            tamper_result = blockchain.verify_consensus_record(
                consensus_id=record_id,
                expected_face_hash=fake_face,
                expected_merkle_root=target_merkle,
            )
            console.print(f"Testing altered face hash: [bold red]{fake_face}[/bold red]")
            console.print(f"Smart contract verifyConsensusRecord() returned: [bold]{tamper_result.is_verified}[/bold]")
            if not tamper_result.is_verified:
                console.print("[bold green]✅ Tamper defense verified! The smart contract successfully rejected the altered consensus data.[/bold green]")

    else:
        # Legacy Single-Record Verification
        total_records = contract.functions.recordCount().call()
        if record_id >= total_records:
            console.print(f"[bold red]❌ Record #{record_id} does not exist! Total in registry: {total_records}[/bold red]")
            raise typer.Exit(code=1)

        raw = contract.functions.getRecord(record_id).call()
        on_chain_face_hex = raw[0].hex()
        on_chain_match_hex = raw[1].hex()

        target_face = expected_face_hash or on_chain_face_hex
        target_match = expected_merkle_root or on_chain_match_hex

        verification = blockchain.verify_record(record_id, target_face, target_match)
        console.print(f"Record #{record_id} Verified: {verification.is_verified}")


if __name__ == "__main__":
    app()
