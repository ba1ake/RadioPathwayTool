from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import discord
from dotenv import load_dotenv

from main import get_propagation_report
from ai_assistant import ask_radio_assistant


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv(
    "DISCORD_BOT_TOKEN"
)

if not DISCORD_BOT_TOKEN:
    raise RuntimeError(
        "DISCORD_BOT_TOKEN is not set. "
        "Add it to your .env file."
    )


# ============================================================
# CACHE
# ============================================================

CACHE_SECONDS = 300

cached_report: Any | None = None
cached_report_time: float = 0.0

report_lock = asyncio.Lock()


# ============================================================
# AI CONVERSATION MEMORY
# ============================================================

# Conversation history is kept separately for each:
#
#     guild + channel + user
#
# This prevents one user's conversation from leaking into
# another user's conversation.
conversation_history: dict[
    tuple[int, int, int],
    list[dict[str, Any]],
] = {}

MAX_HISTORY_MESSAGES = 10


# ============================================================
# DISCORD CLIENT
# ============================================================

intents = discord.Intents.default()

# Required for reading normal Discord messages.
intents.message_content = True

client = discord.Client(
    intents=intents
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def get_value(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Read a value from either a dictionary or an object.

    main.py currently contains a mixture of structured data
    and dictionaries, so this keeps the Discord layer robust.
    """

    if isinstance(obj, dict):
        return obj.get(
            key,
            default,
        )

    return getattr(
        obj,
        key,
        default,
    )


def format_value(
    value: Any,
    default: str = "N/A",
) -> str:
    """
    Convert None/empty values into a Discord-friendly string.
    """

    if value is None:
        return default

    return str(value)


# ============================================================
# PROPAGATION REPORT CACHE
# ============================================================

async def get_cached_report() -> Any:
    """
    Return the current propagation report.

    The expensive report generation runs in a worker thread so
    it does not block the Discord event loop.
    """

    global cached_report
    global cached_report_time

    now = time.monotonic()

    if (
        cached_report is not None
        and now - cached_report_time < CACHE_SECONDS
    ):
        return cached_report

    async with report_lock:

        # Check again after acquiring the lock.
        now = time.monotonic()

        if (
            cached_report is not None
            and now - cached_report_time < CACHE_SECONDS
        ):
            return cached_report

        report = await asyncio.to_thread(
            get_propagation_report
        )

        cached_report = report
        cached_report_time = time.monotonic()

        return report


# ============================================================
# LONG MESSAGE HANDLING
# ============================================================

async def send_long_message(
    destination: Any,
    text: str,
) -> None:
    """
    Discord messages have a 2000-character limit.

    Split long responses at newline boundaries where possible.
    """

    if not text:
        return

    max_length = 1900

    if len(text) <= max_length:
        await destination.send(text)
        return

    remaining = text

    while remaining:

        if len(remaining) <= max_length:
            await destination.send(
                remaining
            )
            break

        split_at = remaining.rfind(
            "\n",
            0,
            max_length,
        )

        if split_at <= 0:
            split_at = max_length

        chunk = remaining[:split_at]

        await destination.send(
            chunk
        )

        remaining = remaining[
            split_at:
        ].lstrip("\n")


# ============================================================
# EMBED HELPERS
# ============================================================

def build_propagation_embed(
    report: Any,
) -> discord.Embed:

    embed = discord.Embed(
        title="📡 RadioPathwayTool",
        description=(
            "Current HF propagation conditions"
        ),
    )

    hap = get_value(
        report,
        "hap",
        {},
    )

    current = get_value(
        hap,
        "current_recommendation",
        None,
    )

    next_transition = get_value(
        hap,
        "next_transition",
        None,
    )

    regional = get_value(
        hap,
        "regional_distribution",
        None,
    )

    if current is not None:
        embed.add_field(
            name="Current HAP",
            value=str(current),
            inline=False,
        )

    if next_transition is not None:
        embed.add_field(
            name="Next transition",
            value=str(next_transition),
            inline=False,
        )

    if regional is not None:
        embed.add_field(
            name="Regional HAP",
            value=str(regional),
            inline=False,
        )

    embed.set_footer(
        text="HAP is a prediction, not a guarantee of contact."
    )

    return embed


def build_ionosphere_embed(
    report: Any,
) -> discord.Embed:

    embed = discord.Embed(
        title="🌐 Ionosphere",
        description=(
            "Current Australian Space Weather Services "
            "ionospheric observations"
        ),
    )

    ionosphere = get_value(
        report,
        "ionosphere",
        [],
    )

    if isinstance(
        ionosphere,
        dict,
    ):
        # Some implementations may return a wrapper dictionary.
        observations = ionosphere.get(
            "observations",
            ionosphere,
        )
    else:
        observations = ionosphere

    lines: list[str] = []

    if isinstance(
        observations,
        dict,
    ):
        iterable = observations.items()
    else:
        iterable = []

        if isinstance(
            observations,
            list,
        ):
            iterable = [
                (
                    get_value(
                        item,
                        "station",
                        "Station",
                    ),
                    item,
                )
                for item in observations
            ]

    for station, observation in iterable:

        condition = get_value(
            observation,
            "condition",
            None,
        )

        percent = get_value(
            observation,
            "percent_difference",
            None,
        )

        if percent is not None:
            lines.append(
                f"**{station}:** "
                f"{condition} ({percent:+.0f}%)"
            )
        else:
            lines.append(
                f"**{station}:** "
                f"{format_value(condition)}"
            )

    if not lines:
        lines.append(
            "No ionospheric observations available."
        )

    text = "\n".join(lines)

    # Keep embed field under Discord's 1024-character limit.
    if len(text) > 1000:
        text = text[:997] + "..."

    embed.add_field(
        name="Observations",
        value=text,
        inline=False,
    )

    return embed


def build_space_weather_embed(
    report: Any,
) -> discord.Embed:

    embed = discord.Embed(
        title="☀️ Space Weather",
        description=(
            "Current solar and geomagnetic conditions"
        ),
    )

    sw = get_value(
        report,
        "space_weather",
        {},
    )

    f107 = get_value(
        sw,
        "solar_flux_10_7",
        None,
    )

    sunspots = get_value(
        sw,
        "sunspot_number",
        None,
    )

    planetary_k = get_value(
        sw,
        "planetary_k_index",
        None,
    )

    australian_k = get_value(
        sw,
        "australian_k_index",
        None,
    )

    a_index = get_value(
        sw,
        "a_index",
        None,
    )

    dst = get_value(
        sw,
        "dst_index",
        None,
    )

    embed.add_field(
        name="Solar",
        value=(
            f"F10.7: {format_value(f107)}\n"
            f"Sunspots: {format_value(sunspots)}"
        ),
        inline=True,
    )

    embed.add_field(
        name="Geomagnetic",
        value=(
            f"Planetary K: {format_value(planetary_k)}\n"
            f"Australian K: {format_value(australian_k)}\n"
            f"A-index: {format_value(a_index)}\n"
            f"Dst: {format_value(dst)}"
        ),
        inline=True,
    )

    alerts = get_value(
        sw,
        "active_alerts",
        [],
    )

    warnings = get_value(
        sw,
        "active_warnings",
        [],
    )

    if alerts:
        lines = []

        for alert in alerts:

            if isinstance(
                alert,
                dict,
            ):
                message = (
                    alert.get("message")
                    or alert.get("text")
                    or str(alert)
                )
            else:
                message = str(alert)

            lines.append(
                f"• {message}"
            )

        alert_text = "\n".join(lines)

        if len(alert_text) > 1000:
            alert_text = (
                alert_text[:997]
                + "..."
            )

        embed.add_field(
            name="Current watches / alerts",
            value=alert_text,
            inline=False,
        )

    if warnings:
        lines = []

        for warning in warnings:

            if isinstance(
                warning,
                dict,
            ):
                message = (
                    warning.get("message")
                    or warning.get("text")
                    or str(warning)
                )
            else:
                message = str(warning)

            lines.append(
                f"• {message}"
            )

        warning_text = "\n".join(lines)

        if len(warning_text) > 1000:
            warning_text = (
                warning_text[:997]
                + "..."
            )

        embed.add_field(
            name="Warnings",
            value=warning_text,
            inline=False,
        )

    return embed


def build_status_embed(
    report: Any,
) -> discord.Embed:

    embed = discord.Embed(
        title="🩺 RadioPathwayTool Status",
    )

    hap = get_value(
        report,
        "hap",
        None,
    )

    ionosphere = get_value(
        report,
        "ionosphere",
        None,
    )

    space_weather = get_value(
        report,
        "space_weather",
        None,
    )

    embed.add_field(
        name="HAP",
        value=(
            "✅ Available"
            if hap is not None
            else "❌ Unavailable"
        ),
        inline=True,
    )

    embed.add_field(
        name="Ionosphere",
        value=(
            "✅ Available"
            if ionosphere is not None
            else "❌ Unavailable"
        ),
        inline=True,
    )

    embed.add_field(
        name="Space weather",
        value=(
            "✅ Available"
            if space_weather is not None
            else "❌ Unavailable"
        ),
        inline=True,
    )

    retrieved = get_value(
        space_weather,
        "retrieved_utc",
        None,
    )

    if retrieved:
        embed.add_field(
            name="Space-weather retrieval",
            value=str(retrieved),
            inline=False,
        )

    return embed


# ============================================================
# HELP
# ============================================================

def build_help_text() -> str:

    return """
**📡 RadioPathwayTool Commands**

`$prop`
Current propagation overview.

`$hap`
Current HAP forecast and regional distribution.

`$ionosphere`
Current ionospheric observations.

`$spaceweather`
Current solar and geomagnetic conditions.

`$status`
Data-source health/status.

`$help`
Show this help.

**🤖 Natural language**

You can also just ask me questions normally.

Examples:

> What's propagation like right now?

> Is 80m worth trying tonight?

> What's happening with the ionosphere?

> Why is 160m being recommended?

> What are the current geomagnetic conditions?

> How might the current conditions affect 20m?

The AI uses the RadioPathwayTool data rather than inventing
current propagation conditions.
""".strip()


# ============================================================
# AI QUESTION HANDLER
# ============================================================

async def handle_ai_question(
    message: discord.Message,
) -> None:

    # Don't respond to bots.
    if message.author.bot:
        return

    # Conversation key.
    #
    # guild_id can be None for DMs, so convert it to 0.
    guild_id = (
        message.guild.id
        if message.guild
        else 0
    )

    channel_id = message.channel.id
    user_id = message.author.id

    conversation_key = (
        guild_id,
        channel_id,
        user_id,
    )

    history = conversation_history.get(
        conversation_key,
        [],
    )

    # Show that the bot is working.
    thinking_message = await message.channel.send(
        "🤖 Thinking..."
    )

    try:

        answer, updated_history = await asyncio.to_thread(
            ask_radio_assistant,
            message.content,
            history,
        )

        conversation_history[
            conversation_key
        ] = updated_history[-MAX_HISTORY_MESSAGES:]

        await thinking_message.delete()

        await send_long_message(
            message.channel,
            answer,
        )

    except Exception as exc:

        print(
            "AI assistant error:",
            repr(exc),
        )

        error_text = (
            "⚠️ I couldn't process that question "
            "right now.\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )

        try:
            await thinking_message.edit(
                content=error_text
            )
        except Exception:
            await send_long_message(
                message.channel,
                error_text,
            )


# ============================================================
# READY EVENT
# ============================================================

@client.event
async def on_ready():

    print(
        f"Logged in as "
        f"{client.user} "
        f"(ID: {client.user.id})"
    )

    print(
        f"Connected to {len(client.guilds)} guild(s)."
    )

    activity = discord.Game(
        name="$help | HF propagation"
    )

    await client.change_presence(
        activity=activity
    )


# ============================================================
# MESSAGE EVENT
# ============================================================

@client.event
async def on_message(
    message: discord.Message,
):

    # Never respond to ourselves or other bots.
    if message.author.bot:
        return

    content = message.content.strip()

    if not content:
        return

    lower = content.lower()

    # ========================================================
    # BASIC COMMANDS
    # ========================================================

    if lower in {
        "$hello",
        "$hi",
    }:

        await message.channel.send(
            f"Hello {message.author.mention}! "
            "📡 Ask me a propagation question or use `$help`."
        )

        return

    # ========================================================
    # HELP
    # ========================================================

    if lower in {
        "$help",
        "$commands",
    }:

        await message.channel.send(
            build_help_text()
        )

        return

    # ========================================================
    # PROPAGATION
    # ========================================================

    if lower in {
        "$prop",
        "$propagation",
        "$radio",
        "$conditions",
    }:

        try:

            report = await get_cached_report()

            embed = build_propagation_embed(
                report
            )

            await message.channel.send(
                embed=embed
            )

        except Exception as exc:

            print(
                "Propagation command error:",
                repr(exc),
            )

            await message.channel.send(
                "❌ Failed to retrieve propagation data."
            )

        return

    # ========================================================
    # HAP
    # ========================================================

    if lower in {
        "$hap",
        "$forecast",
    }:

        try:

            report = await get_cached_report()

            embed = build_propagation_embed(
                report
            )

            await message.channel.send(
                embed=embed
            )

        except Exception as exc:

            print(
                "HAP command error:",
                repr(exc),
            )

            await message.channel.send(
                "❌ Failed to retrieve HAP data."
            )

        return

    # ========================================================
    # IONOSPHERE
    # ========================================================

    if lower in {
        "$ionosphere",
        "$iono",
    }:

        try:

            report = await get_cached_report()

            embed = build_ionosphere_embed(
                report
            )

            await message.channel.send(
                embed=embed
            )

        except Exception as exc:

            print(
                "Ionosphere command error:",
                repr(exc),
            )

            await message.channel.send(
                "❌ Failed to retrieve ionosphere data."
            )

        return

    # ========================================================
    # SPACE WEATHER
    # ========================================================

    if lower in {
        "$spaceweather",
        "$space",
        "$solar",
    }:

        try:

            report = await get_cached_report()

            embed = build_space_weather_embed(
                report
            )

            await message.channel.send(
                embed=embed
            )

        except Exception as exc:

            print(
                "Space-weather command error:",
                repr(exc),
            )

            await message.channel.send(
                "❌ Failed to retrieve space-weather data."
            )

        return

    # ========================================================
    # STATUS
    # ========================================================

    if lower in {
        "$status",
        "$health",
    }:

        try:

            report = await get_cached_report()

            embed = build_status_embed(
                report
            )

            await message.channel.send(
                embed=embed
            )

        except Exception as exc:

            print(
                "Status command error:",
                repr(exc),
            )

            await message.channel.send(
                "❌ Failed to retrieve system status."
            )

        return

    # ========================================================
    # UNKNOWN $ COMMAND
    # ========================================================

    if content.startswith("$"):

        await message.channel.send(
            "❓ Unknown command. "
            "Use `$help` to see available commands, "
            "or ask me a normal question."
        )

        return

    # ========================================================
    # NATURAL LANGUAGE
    # ========================================================

    await handle_ai_question(
        message
    )


# ============================================================
# START BOT
# ============================================================

if __name__ == "__main__":

    client.run(
        DISCORD_BOT_TOKEN
    )