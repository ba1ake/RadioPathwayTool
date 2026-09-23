import asyncio
import os
import time

import discord
from dotenv import load_dotenv

from main import get_propagation_report


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")

CACHE_SECONDS = 300  # 5 minutes


# ============================================================
# TOKEN CHECK
# ============================================================

if not TOKEN:
    raise RuntimeError(
        "DISCORD_BOT_TOKEN is not set. "
        "Check your .env file."
    )


# ============================================================
# DISCORD SETUP
# ============================================================

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


# ============================================================
# REPORT CACHE
# ============================================================

cached_report = None
cached_report_time = 0

report_lock = asyncio.Lock()


async def get_cached_report():
    """
    Get a propagation report.

    Reports are cached for CACHE_SECONDS so multiple Discord
    commands don't repeatedly hammer the HAP/SWS services.
    """

    global cached_report
    global cached_report_time

    now = time.time()

    # --------------------------------------------------------
    # Return cached report if still valid
    # --------------------------------------------------------

    if (
        cached_report is not None
        and now - cached_report_time < CACHE_SECONDS
    ):
        return cached_report

    # --------------------------------------------------------
    # Prevent multiple simultaneous reports
    # --------------------------------------------------------

    async with report_lock:

        now = time.time()

        if (
            cached_report is not None
            and now - cached_report_time < CACHE_SECONDS
        ):
            return cached_report

        # ----------------------------------------------------
        # Run blocking propagation code in another thread
        # ----------------------------------------------------

        report = await asyncio.to_thread(
            get_propagation_report
        )

        cached_report = report
        cached_report_time = time.time()

        return report


# ============================================================
# MESSAGE HELPER
# ============================================================

async def send_long_message(
    channel,
    text,
    limit=1900,
):
    """
    Discord has a 2000 character message limit.

    Split long text into multiple messages.
    """

    if len(text) <= limit:

        await channel.send(text)
        return

    while text:

        chunk = text[:limit]

        # Try to split at a newline
        split_at = chunk.rfind("\n")

        if split_at > 500:

            chunk = chunk[:split_at]

        await channel.send(chunk)

        text = text[len(chunk):]


# ============================================================
# PROPAGATION EMBED
# ============================================================

def build_propagation_embed(report):

    embed = discord.Embed(
        title="📡 HF Propagation",
        description=(
            f"**{report.location_name}**\n"
            f"{report.generated_local.strftime('%d %b %Y %H:%M')} NZ"
        ),
    )

    # --------------------------------------------------------
    # HAP
    # --------------------------------------------------------

    if (
        report.data_status
        and report.data_status.hap_available
    ):

        if report.current_band:

            if report.current_frequency_mhz:

                current_text = (
                    f"**{report.current_band}** "
                    f"({report.current_frequency_mhz:.3f} MHz)"
                )

            else:

                current_text = (
                    f"**{report.current_band}**"
                )

            if report.current_support is not None:

                current_text += (
                    f"\nRegional support: "
                    f"{report.current_support}"
                )

        else:

            current_text = "No current HAP recommendation."

        # ----------------------------------------------------
        # Next transition
        # ----------------------------------------------------

        if report.next_transition_utc is not None:

            if report.next_transition_frequency_mhz:

                next_text = (
                    f"{report.next_transition_utc:02d} UTC → "
                    f"**{report.next_transition_band}** "
                    f"({report.next_transition_frequency_mhz:.3f} MHz)"
                )

            else:

                next_text = (
                    f"{report.next_transition_utc:02d} UTC → "
                    f"**{report.next_transition_band}**"
                )

            current_text += (
                f"\nNext change: {next_text}"
            )

        embed.add_field(
            name="🛰️ HAP",
            value=current_text,
            inline=False,
        )

    else:

        hap_error = (
            report.data_status.hap_error
            if report.data_status
            else "Unknown error"
        )

        embed.add_field(
            name="🛰️ HAP",
            value=(
                "❌ HAP data unavailable.\n"
                f"```text\n{hap_error}\n```"
            ),
            inline=False,
        )

    # --------------------------------------------------------
    # Ionosphere
    # --------------------------------------------------------

    if (
        report.data_status
        and report.data_status.ionosphere_available
    ):

        observations = (
            report.ionosphere_observations
            or {}
        )

        # Prioritise useful non-normal observations
        interesting = []

        for observation in observations.values():

            condition = getattr(
                observation,
                "condition",
                "",
            )

            percent = getattr(
                observation,
                "percent_difference",
                None,
            )

            station_name = getattr(
                observation,
                "station_name",
                "Unknown",
            )

            if (
                percent is not None
                and abs(percent) >= 10
            ):

                if percent > 0:

                    text = (
                        f"{station_name}: "
                        f"{condition} "
                        f"(+{percent:.0f}%)"
                    )

                else:

                    text = (
                        f"{station_name}: "
                        f"{condition} "
                        f"({percent:.0f}%)"
                    )

                interesting.append(text)

        if interesting:

            iono_text = "\n".join(
                interesting[:8]
            )

        else:

            iono_text = (
                "No significant deviations reported."
            )

    else:

        error = (
            report.data_status.ionosphere_error
            if report.data_status
            else "Unknown error"
        )

        iono_text = (
            "❌ Ionosphere data unavailable.\n"
            f"`{error}`"
        )

    embed.add_field(
        name="🌐 Ionosphere",
        value=iono_text,
        inline=False,
    )

    # --------------------------------------------------------
    # Space Weather
    # --------------------------------------------------------

    if (
        report.data_status
        and report.data_status.space_weather_available
    ):

        sw = report.space_weather

        lines = []

        solar_flux = getattr(
            sw,
            "solar_flux_10_7",
            None,
        )

        sunspots = getattr(
            sw,
            "sunspot_number",
            None,
        )

        planetary_k = getattr(
            sw,
            "planetary_k_index",
            None,
        )

        australian_k = getattr(
            sw,
            "australian_k_index",
            None,
        )

        if solar_flux is not None:

            lines.append(
                f"Solar flux: **{solar_flux}**"
            )

        if sunspots is not None:

            lines.append(
                f"Sunspots: **{sunspots}**"
            )

        if planetary_k is not None:

            lines.append(
                f"Planetary K: **{planetary_k}**"
            )

        if australian_k is not None:

            lines.append(
                f"Australian K: **{australian_k}**"
            )

        if not lines:

            lines.append(
                "No summary values available."
            )

        space_text = "\n".join(lines)

    else:

        error = (
            report.data_status.space_weather_error
            if report.data_status
            else "Unknown error"
        )

        space_text = (
            "❌ Space weather unavailable.\n"
            f"`{error}`"
        )

    embed.add_field(
        name="☀️ Space Weather",
        value=space_text,
        inline=False,
    )

    return embed


# ============================================================
# IONOSPHERE EMBED
# ============================================================

def build_ionosphere_embed(report):

    embed = discord.Embed(
        title="🌐 Ionosphere",
        description=(
            f"Australian SWS observations\n"
            f"{report.generated_local.strftime('%d %b %Y %H:%M')} NZ"
        ),
    )

    if not (
        report.data_status
        and report.data_status.ionosphere_available
    ):

        error = (
            report.data_status.ionosphere_error
            if report.data_status
            else "Unknown error"
        )

        embed.description += (
            f"\n\n❌ Data unavailable.\n"
            f"```text\n{error}\n```"
        )

        return embed

    observations = (
        report.ionosphere_observations
        or {}
    )

    lines = []

    for observation in observations.values():

        station_name = getattr(
            observation,
            "station_name",
            "Unknown",
        )

        condition = getattr(
            observation,
            "condition",
            "unknown",
        )

        percent = getattr(
            observation,
            "percent_difference",
            None,
        )

        if percent is not None:

            if percent > 0:

                condition_text = (
                    f"{condition} "
                    f"(+{percent:.0f}%)"
                )

            else:

                condition_text = (
                    f"{condition} "
                    f"({percent:.0f}%)"
                )

        else:

            condition_text = condition

        lines.append(
            f"**{station_name}** — {condition_text}"
        )

    if not lines:

        lines.append(
            "No observations available."
        )

    # Discord embed field limit
    text = "\n".join(lines)

    if len(text) > 1024:

        text = text[:1000] + "..."

    embed.add_field(
        name="Stations",
        value=text,
        inline=False,
    )

    return embed


# ============================================================
# SPACE WEATHER EMBED
# ============================================================

def build_space_weather_embed(report):

    embed = discord.Embed(
        title="☀️ Space Weather",
        description=(
            f"{report.generated_local.strftime('%d %b %Y %H:%M')} NZ"
        ),
    )

    if not (
        report.data_status
        and report.data_status.space_weather_available
    ):

        error = (
            report.data_status.space_weather_error
            if report.data_status
            else "Unknown error"
        )

        embed.add_field(
            name="Status",
            value=(
                "❌ Data unavailable.\n"
                f"```text\n{error}\n```"
            ),
            inline=False,
        )

        return embed

    sw = report.space_weather

    fields = [
        (
            "Solar Flux (F10.7)",
            getattr(
                sw,
                "solar_flux_10_7",
                None,
            ),
        ),
        (
            "Sunspot Number",
            getattr(
                sw,
                "sunspot_number",
                None,
            ),
        ),
        (
            "Planetary K",
            getattr(
                sw,
                "planetary_k_index",
                None,
            ),
        ),
        (
            "Australian K",
            getattr(
                sw,
                "australian_k_index",
                None,
            ),
        ),
        (
            "A-index",
            getattr(
                sw,
                "a_index",
                None,
            ),
        ),
        (
            "Dst",
            getattr(
                sw,
                "dst_index",
                None,
            ),
        ),
    ]

    for name, value in fields:

        if value is not None:

            embed.add_field(
                name=name,
                value=str(value),
                inline=True,
            )

    # --------------------------------------------------------
    # Alerts
    # --------------------------------------------------------

    alerts = getattr(
        sw,
        "active_alerts",
        None,
    )

    warnings = getattr(
        sw,
        "active_warnings",
        None,
    )

    if alerts:

        embed.add_field(
            name="🚨 Alerts",
            value="\n".join(
                str(x) for x in alerts
            )[:1024],
            inline=False,
        )

    if warnings:

        embed.add_field(
            name="⚠️ Warnings",
            value="\n".join(
                str(x) for x in warnings
            )[:1024],
            inline=False,
        )

    return embed


# ============================================================
# STATUS EMBED
# ============================================================

def build_status_embed(report):

    status = report.data_status

    embed = discord.Embed(
        title="🩺 RadioPathwayTool Status",
        description=(
            f"Location: **{report.location_name}**\n"
            f"Generated: "
            f"{report.generated_local.strftime('%d %b %Y %H:%M')} NZ"
        ),
    )

    if status is None:

        embed.add_field(
            name="Status",
            value="❓ No status information.",
            inline=False,
        )

        return embed

    # --------------------------------------------------------
    # HAP
    # --------------------------------------------------------

    if status.hap_available:

        hap_status = "🟢 Available"

    else:

        hap_status = "🔴 Unavailable"

    # --------------------------------------------------------
    # Ionosphere
    # --------------------------------------------------------

    if status.ionosphere_available:

        iono_status = "🟢 Available"

    else:

        iono_status = "🔴 Unavailable"

    # --------------------------------------------------------
    # Space Weather
    # --------------------------------------------------------

    if status.space_weather_available:

        sw_status = "🟢 Available"

    else:

        sw_status = "🔴 Unavailable"

    embed.add_field(
        name="🛰️ HAP",
        value=hap_status,
        inline=True,
    )

    embed.add_field(
        name="🌐 Ionosphere",
        value=iono_status,
        inline=True,
    )

    embed.add_field(
        name="☀️ Space Weather",
        value=sw_status,
        inline=True,
    )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    errors = []

    if status.hap_error:

        errors.append(
            f"HAP: {status.hap_error}"
        )

    if status.ionosphere_error:

        errors.append(
            f"Ionosphere: {status.ionosphere_error}"
        )

    if status.space_weather_error:

        errors.append(
            f"Space weather: {status.space_weather_error}"
        )

    if errors:

        embed.add_field(
            name="Errors",
            value="\n".join(errors)[:1024],
            inline=False,
        )

    return embed


# ============================================================
# BOT READY
# ============================================================

@client.event
async def on_ready():

    print(
        f"We have logged in as "
        f"{client.user}"
    )

    print(
        f"Bot ID: {client.user.id}"
    )

    print(
        f"Connected to {len(client.guilds)} guild(s)"
    )

    await client.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="$help | HF propagation",
        )
    )


# ============================================================
# MESSAGE HANDLER
# ============================================================

@client.event
async def on_message(message):

    # Ignore ourselves
    if message.author == client.user:
        return

    content = message.content.strip()

    # ========================================================
    # HELLO
    # ========================================================

    if content.lower() in (
        "$hello",
        "$hi",
    ):

        await message.channel.send(
            f"Hello {message.author.mention}! 👋"
        )

        return

    # ========================================================
    # HELP
    # ========================================================

    if content.lower() in (
        "$help",
        "$commands",
    ):

        help_text = (
            "**📡 RadioPathwayTool Commands**\n\n"

            "`$prop`\n"
            "Current HF propagation report.\n\n"

            "`$hap`\n"
            "Current HAP propagation recommendation.\n\n"

            "`$ionosphere`\n"
            "Current Australian SWS ionosphere observations.\n\n"

            "`$spaceweather`\n"
            "Current solar and geomagnetic conditions.\n\n"

            "`$status`\n"
            "Show data-source availability and errors.\n\n"

            "`$hello`\n"
            "Say hello to the bot.\n"
        )

        await send_long_message(
            message.channel,
            help_text,
        )

        return

    # ========================================================
    # PROPAGATION
    # ========================================================

    if content.lower() in (
        "$prop",
        "$propagation",
        "$radio",
        "$conditions",
    ):

        status_message = await message.channel.send(
            "📡 Collecting propagation data..."
        )

        try:

            report = await get_cached_report()

            embed = build_propagation_embed(
                report
            )

            await status_message.edit(
                content=None,
                embed=embed,
            )

        except Exception as exc:

            await status_message.edit(
                content=(
                    "❌ Failed to generate propagation report.\n"
                    f"```text\n"
                    f"{type(exc).__name__}: {exc}"
                    f"\n```"
                )
            )

        return

    # ========================================================
    # HAP
    # ========================================================

    if content.lower() in (
        "$hap",
        "$forecast",
    ):

        status_message = await message.channel.send(
            "🛰️ Collecting HAP forecast..."
        )

        try:

            report = await get_cached_report()

            if not (
                report.data_status
                and report.data_status.hap_available
            ):

                error = (
                    report.data_status.hap_error
                    if report.data_status
                    else "Unknown error"
                )

                await status_message.edit(
                    content=(
                        "❌ HAP data is currently unavailable.\n"
                        f"```text\n"
                        f"{error}"
                        f"\n```"
                    )
                )

                return

            embed = build_propagation_embed(
                report
            )

            await status_message.edit(
                content=None,
                embed=embed,
            )

        except Exception as exc:

            await status_message.edit(
                content=(
                    "❌ Failed to collect HAP data.\n"
                    f"```text\n"
                    f"{type(exc).__name__}: {exc}"
                    f"\n```"
                )
            )

        return

    # ========================================================
    # IONOSPHERE
    # ========================================================

    if content.lower() in (
        "$ionosphere",
        "$iono",
    ):

        status_message = await message.channel.send(
            "🌐 Collecting ionosphere data..."
        )

        try:

            report = await get_cached_report()

            embed = build_ionosphere_embed(
                report
            )

            await status_message.edit(
                content=None,
                embed=embed,
            )

        except Exception as exc:

            await status_message.edit(
                content=(
                    "❌ Failed to collect ionosphere data.\n"
                    f"```text\n"
                    f"{type(exc).__name__}: {exc}"
                    f"\n```"
                )
            )

        return

    # ========================================================
    # SPACE WEATHER
    # ========================================================

    if content.lower() in (
        "$spaceweather",
        "$space",
        "$solar",
    ):

        status_message = await message.channel.send(
            "☀️ Collecting space weather data..."
        )

        try:

            report = await get_cached_report()

            embed = build_space_weather_embed(
                report
            )

            await status_message.edit(
                content=None,
                embed=embed,
            )

        except Exception as exc:

            await status_message.edit(
                content=(
                    "❌ Failed to collect space weather data.\n"
                    f"```text\n"
                    f"{type(exc).__name__}: {exc}"
                    f"\n```"
                )
            )

        return

    # ========================================================
    # STATUS
    # ========================================================

    if content.lower() in (
        "$status",
        "$health",
    ):

        status_message = await message.channel.send(
            "🩺 Checking RadioPathwayTool..."
        )

        try:

            report = await get_cached_report()

            embed = build_status_embed(
                report
            )

            await status_message.edit(
                content=None,
                embed=embed,
            )

        except Exception as exc:

            await status_message.edit(
                content=(
                    "❌ Failed to check system status.\n"
                    f"```text\n"
                    f"{type(exc).__name__}: {exc}"
                    f"\n```"
                )
            )

        return

    # ========================================================
    # UNKNOWN COMMAND
    # ========================================================

    if content.startswith("$"):

        await message.channel.send(
            "❓ Unknown command. "
            "Use `$help` to see available commands."
        )


# ============================================================
# START BOT
# ============================================================

client.run(TOKEN)