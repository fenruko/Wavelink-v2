"""
MIT License

Copyright (c) 2019-Current PythonistaGuild, EvieePy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import discord
from discord.abc import Connectable
from discord.utils import MISSING

import wavelink

from .enums import AutoPlayMode, NodeStatus, QueueMode
from .exceptions import (
    ChannelTimeoutException,
    InvalidChannelStateException,
    InvalidNodeException,
    LavalinkException,
    LavalinkLoadException,
    QueueEmpty,
)
from .filters import Filters
from .node import Pool
from .payloads import (
    PlayerUpdateEventPayload,
    TrackEndEventPayload,
    TrackStartEventPayload,
)
from .queue import Queue
from .tracks import Playable, Playlist


if TYPE_CHECKING:
    from discord.abc import Connectable
    from discord.types.voice import (
        GuildVoiceState as GuildVoiceStatePayload,
        VoiceServerUpdate as VoiceServerUpdatePayload,
    )
    from typing_extensions import Self

    from .node import Node
    from .payloads import (
        PlayerUpdateEventPayload,
        TrackEndEventPayload,
        TrackStartEventPayload,
    )
    from .types.request import Request as RequestPayload
    from .types.state import PlayerBasicState, PlayerVoiceState, VoiceState

    VocalGuildChannel = discord.VoiceChannel | discord.StageChannel

logger: logging.Logger = logging.getLogger(__name__)


# AutoPlay only reads the tail of auto_queue.history. Cap it so a 24/7 player
# does not retain every recommended track it has ever played.
_AUTO_HISTORY_LIMIT = 512


class Player(discord.VoiceProtocol):
    """The Player is a :class:`discord.VoiceProtocol` used to connect your :class:`discord.Client` to a
    :class:`discord.VoiceChannel`.

    The player controls the music elements of the bot including playing tracks, the queue, connecting etc.
    See Also: The various methods available.

    .. note::

        Since the Player is a :class:`discord.VoiceProtocol`, it is attached to the various ``voice_client`` attributes
        in discord.py, including ``guild.voice_client``, ``ctx.voice_client`` and ``interaction.voice_client``.

    Attributes
    ----------
    queue: :class:`~wavelink.Queue`
        The queue associated with this player.
    auto_queue: :class:`~wavelink.Queue`
        The auto_queue associated with this player. This queue holds tracks that are recommended by the AutoPlay feature.
    """

    channel: VocalGuildChannel

    def __call__(self, client: discord.Client, channel: VocalGuildChannel) -> Self:
        super().__init__(client, channel)

        self._guild = channel.guild

        return self

    def __init__(
        self, client: discord.Client = MISSING, channel: Connectable = MISSING, *, nodes: list[Node] | None = None
    ) -> None:
        super().__init__(client, channel)

        self.client: discord.Client = client
        self._guild: discord.Guild | None = None

        self._voice_state: PlayerVoiceState = {"voice": {}}

        self._node: Node
        if not nodes:
            self._node = Pool.get_node()
        else:
            self._node = min(nodes, key=lambda n: len(n._players))

        if self.client is MISSING and self.node.client:
            self.client = self.node.client

        self._last_update: int | None = None
        self._last_position: int = 0
        self._ping: int = -1

        self._connected: bool = False
        self._connection_event: asyncio.Event = asyncio.Event()

        self._current: Playable | None = None
        self._original: Playable | None = None
        self._previous: Playable | None = None

        self.queue: Queue = Queue()
        self.auto_queue: Queue = Queue()

        self._volume: int = 100
        self._paused: bool = False

        self._auto_cutoff: int = 20
        self._auto_weight: int = 3
        self._previous_seeds_cutoff: int = self._auto_cutoff * self._auto_weight
        self._history_count: int | None = None

        self._autoplay: AutoPlayMode = AutoPlayMode.disabled
        # Bounded ring of recent recommendation seeds. asyncio.Queue was a
        # synchronization primitive used only as a maxlen buffer.
        self._previous_seeds: deque[str] = deque(maxlen=self._previous_seeds_cutoff)

        self._auto_lock: asyncio.Lock = asyncio.Lock()
        self._recommend_lock: asyncio.Lock = asyncio.Lock()
        self._recommend_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._error_count: int = 0

        self._inactive_channel_limit: int | None = self._node._inactive_channel_tokens
        self._inactive_channel_count: int = self._inactive_channel_limit if self._inactive_channel_limit else 0

        self._filters: Filters = Filters()

        # call_later handle. A Task + sleep per track-end was a coroutine and a
        # done-callback for what is just a timer.
        self._inactivity_handle: asyncio.TimerHandle | None = None
        self._inactivity_wait: int | None = self._node._inactive_player_timeout

        self._should_wait: int = 10
        self._reconnecting: asyncio.Event = asyncio.Event()
        self._reconnecting.set()

    async def _disconnected_wait(self, code: int, by_remote: bool) -> None:
        if code != 4014 or not by_remote:
            return

        self._connected = False
        await self._reconnecting.wait()

        if self._connected:
            return

        await self._destroy()

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _on_inactive(self) -> None:
        self._inactivity_handle = None

        guild = self._guild
        if guild is None or self.playing:
            logger.debug("Disregarding inactivity check for player %s.", getattr(guild, "id", None))
            return

        self.client.dispatch("wavelink_inactive_player", self)
        logger.debug('Dispatched "on_wavelink_inactive_player" for Player <%s>.', guild.id)

    def _inactivity_cancel(self) -> None:
        handle = self._inactivity_handle
        if handle is not None:
            handle.cancel()
            self._inactivity_handle = None

    def _inactivity_start(self) -> None:
        wait = self._inactivity_wait
        if not wait or wait <= 0:
            return

        self._inactivity_cancel()
        self._inactivity_handle = asyncio.get_running_loop().call_later(wait, self._on_inactive)

    def _track_start(self, payload: TrackStartEventPayload) -> None:
        self._inactivity_cancel()

    async def _auto_play_event(self, payload: TrackEndEventPayload) -> None:
        if not self.channel:
            return

        has_listeners = any(not member.bot for member in self.channel.members)
        self._inactive_channel_count = (
            self._inactive_channel_count - 1 if not has_listeners else self._inactive_channel_limit or 0
        )

        if self._inactive_channel_limit and self._inactive_channel_count <= 0:
            self._inactive_channel_count = self._inactive_channel_limit  # Reset...

            self._inactivity_cancel()
            self.client.dispatch("wavelink_inactive_player", self)

        elif self._autoplay is AutoPlayMode.disabled:
            self._inactivity_start()
            return

        if self._error_count >= 3:
            logger.warning(
                "AutoPlay was unable to continue as you have received too many consecutive errors."
                "Please check the error log on Lavalink."
            )
            self._inactivity_start()
            return

        if payload.reason == "replaced":
            self._error_count = 0
            return

        elif payload.reason == "loadFailed":
            self._error_count += 1

        else:
            self._error_count = 0

        if self.node.status is not NodeStatus.CONNECTED:
            logger.warning(
                '"Unable to use AutoPlay on Player for Guild "%s" due to disconnected Node.', str(self.guild)
            )
            return

        if not isinstance(self.queue, Queue) or not isinstance(self.auto_queue, Queue):  # type: ignore
            logger.warning(
                '"Unable to use AutoPlay on Player for Guild "%s" due to unsupported Queue.', str(self.guild)
            )
            self._inactivity_start()
            return

        if self.queue.mode is QueueMode.loop:
            await self._do_partial(history=False)

        elif self.queue.mode is QueueMode.loop_all or (self._autoplay is AutoPlayMode.partial or self.queue):
            await self._do_partial()

        elif self._autoplay is AutoPlayMode.enabled:
            async with self._auto_lock:
                await self._do_recommendation()

    async def _do_partial(self, *, history: bool = True) -> None:
        # Arm inactivity only when nothing will actually start. A timer created
        # before a successful play was always cancelled by track-start.
        if self._current is not None:
            self._inactivity_start()
            return

        try:
            track: Playable = self.queue.get()
        except QueueEmpty:
            self._inactivity_start()
            return

        try:
            await self.play(track, add_history=history)
        except Exception:
            self._inactivity_start()
            raise

    def _recent(self, queue: Queue, limit: int, *, reverse: bool) -> list[Playable]:
        """Return up to *limit* tracks without copying the rest of the queue."""
        items = queue._items
        length = len(items)
        if length == 0 or limit <= 0:
            return []

        take = limit if limit < length else length
        if not reverse:
            # Walk from the left once. Indexing a deque in a loop is quadratic.
            picked: list[Playable] = []
            for track in items:
                picked.append(track)
                if len(picked) == take:
                    break
            return picked

        start = length - 1
        return [items[i] for i in range(start, start - take, -1)]

    def _push_auto_history(self, track: Playable) -> None:
        history = self.auto_queue.history
        if history is None:
            return

        history.put(track)
        overflow = len(history) - _AUTO_HISTORY_LIMIT
        if overflow > 0:
            items = history._items
            for _ in range(overflow):
                items.popleft()

    def _recommendation_seen(self) -> tuple[set[str], set[str]]:
        """Identifier and encoded sets equivalent to ``track in history_window``.

        ``Playable.__eq__`` is true when either encoded or identifier matches.
        Hash lookups replace a linear scan of long encoded strings.
        """
        identifiers: set[str] = set()
        encoded: set[str] = set()

        windows: list[list[Playable]] = [
            self._recent(self.auto_queue, 40, reverse=False),
            self._recent(self.queue, 40, reverse=False),
        ]
        if self.queue.history is not None:
            windows.append(self._recent(self.queue.history, 40, reverse=True))
        if self.auto_queue.history is not None:
            windows.append(self._recent(self.auto_queue.history, 60, reverse=True))

        for window in windows:
            for track in window:
                identifiers.add(track.identifier)
                encoded.add(track.encoded)

        return identifiers, encoded

    def _build_recommendation_queries(self, populate_track: Playable | None = None) -> tuple[str | None, str | None]:
        assert self.queue.history is not None

        weight = self._auto_weight
        history_limit = max(5, 5 * weight)
        upcoming_limit = max(3, int((5 * weight) / 3))

        previous = self._previous_seeds
        seeds: list[Playable] = [
            track
            for track in self._recent(self.queue.history, history_limit, reverse=True)
            if track.identifier not in previous
        ]
        seeds.extend(
            track
            for track in self._recent(self.auto_queue, upcoming_limit, reverse=False)
            if track.identifier not in previous
        )
        seeds.extend(
            track
            for track in (self._current, self._previous)
            if track is not None and track.identifier not in previous
        )

        random.shuffle(seeds)
        if populate_track is not None:
            seeds.insert(0, populate_track)

        spotify: list[str] = []
        youtube: list[str] = []
        for track in seeds:
            source = track.source
            if source == "spotify":
                spotify.append(track.identifier)
            elif source == "youtube":
                youtube.append(track.identifier)

        count = len(self.queue.history)
        changed_by = min(3, count) if self._history_count is None else count - self._history_count
        if changed_by > 0:
            self._history_count = count

        added = 0
        for track in self._recent(self.queue.history, min(changed_by, 3), reverse=True):
            if added == 2 and track.source == "spotify":
                break

            if track.source == "spotify":
                spotify.insert(0, track.identifier)
                added += 1
            elif track.source == "youtube":
                if youtube:
                    youtube[0] = track.identifier
                else:
                    youtube.append(track.identifier)

        spotify_query: str | None = None
        youtube_query: str | None = None

        if spotify:
            spotify_seeds = spotify[:3]
            spotify_query = f"sprec:seed_tracks={','.join(spotify_seeds)}&limit=10"
            for seed in spotify_seeds:
                self._add_to_previous_seeds(seed)

        if youtube:
            ytm_seed = youtube[0]
            youtube_query = f"https://music.youtube.com/watch?v={ytm_seed}8&list=RD{ytm_seed}"
            self._add_to_previous_seeds(ytm_seed)

        return spotify_query, youtube_query

    async def _search_recommendations(self, query: str | None) -> list[Playable]:
        if not query:
            return []

        try:
            search: wavelink.Search = await Pool.fetch_tracks(query, node=self._node)
        except (LavalinkLoadException, LavalinkException):
            return []

        if not search:
            return []

        if isinstance(search, Playlist):
            return search.tracks

        return search

    async def _apply_recommendations(self, queries: tuple[str | None, str | None], max_population: int) -> int:
        assert self.guild is not None

        spotify_query, youtube_query = queries
        if spotify_query is None and youtube_query is None:
            return 0

        async with self._recommend_lock:
            spotify_tracks, youtube_tracks = await asyncio.gather(
                self._search_recommendations(spotify_query),
                self._search_recommendations(youtube_query),
            )

            filtered = spotify_tracks + youtube_tracks
            if not filtered:
                # The caller logs and arms inactivity when nothing is left to pla     logger.info('Player "%s" could not load any songs via AutoPlay.', self.guild.id)
                return 0

            seen_ids, seen_encoded = self._recommendation_seen()
            random.shuffle(filtered)

            accepted: list[Playable] = []
            for track in filtered:
                if track.identifier in seen_ids or track.encoded in seen_encoded:
                    continue

                track._recommended = True
                accepted.append(track)
                seen_ids.add(track.identifier)
                seen_encoded.add(track.encoded)
                if len(accepted) >= max_population:
                    break

            added = self.auto_queue.put(accepted) if accepted else 0

        logger.debug('Player "%s" added "%s" tracks to the auto_queue via AutoPlay.', self.guild.id, added)
        return added

    def _schedule_recommendation(self, queries: tuple[str | None, str | None], max_population: int) -> None:
        current = self._recommend_task
        if current is not None and not current.done():
            return

        self._recommend_task = self._spawn(self._recommendation_fill(queries, max_population))

    async def _recommendation_fill(self, queries: tuple[str | None, str | None], max_population: int) -> None:
        try:
            await self._apply_recommendations(queries, max_population)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("AutoPlay refill failed for guild %s", getattr(self.guild, "id", None), exc_info=True)

    async def _do_recommendation(
        self,
        *,
        populate_track: wavelink.Playable | None = None,
        max_population: int | None = None,
    ) -> None:
        assert self.guild is not None
        assert self.queue.history is not None and self.auto_queue.history is not None

        max_population_: int = max_population if max_population else self._auto_cutoff

        # A queued recommendation should start immediately. The old path blocked
        # the gap between songs on two Lavalink searches whenever the buffer
        # dropped to the cutoff, even though those searches only append.
        if populate_track is None and self.auto_queue:
            needs_fill = len(self.auto_queue) <= self._auto_cutoff + 1
            queries = self._build_recommendation_queries() if needs_fill else None

            track = self.auto_queue.get()
            self._push_auto_history(track)

            try:
                await self.play(track, add_history=False)
            except Exception:
                self._inactivity_start()
                raise

            if queries not in (None, (None, None)):
                self._schedule_recommendation(queries, max_population_)
            return

        queries = self._build_recommendation_queries(populate_track)
        await self._apply_recommendations(queries, max_population_)

        if self._current is not None or populate_track is not None:
            return

        try:
            now = self.auto_queue.get()
        except QueueEmpty:
            logger.info('Player "%s" could not load any songs via AutoPlay.', self.guild.id)
            self._inactivity_start()
            return

        self._push_auto_history(now)
        try:
            await self.play(now, add_history=False)
        except Exception:
            self._inactivity_start()
            raise

    @property
    def state(self) -> PlayerBasicState:
        """Property returning a dict of the current basic state of the player.

        This property includes the ``voice_state`` received via Discord.

        Returns
        -------
        PlayerBasicState

        .. versionadded:: 3.5.0
        """
        data: PlayerBasicState = {
            "voice_state": self._voice_state.copy(),
            "position": self.position,
            "connected": self.connected,
            "current": self.current,
            "paused": self.paused,
            "volume": self.volume,
            "filters": self.filters,
        }
        return data

    async def switch_node(self, new_node: wavelink.Node, /) -> None:
        """Method which attempts to switch the current node of the player.

        This method initiates a live switch, and all player state will be moved from the current node to the provided
        node.

        .. warning::

            Caution should be used when using this method. If this method fails, your player might be left in a stale
            state. Consider handling cases where the player is unable to connect to the new node. To avoid stale state
            in both wavelink and discord.py, it is recommended to disconnect the player when a RuntimeError occurs.

        Parameters
        ----------
        new_node: :class:`wavelink.Node`
            A positional only argument of a :class:`wavelink.Node`, which is the new node the player will attempt to
            switch to. This must not be the same as the current node.

        Raises
        ------
        InvalidNodeException
            The provided node was identical to the players current node.
        RuntimeError
            The player was unable to connect properly to the new node. At this point your player might be in a stale
            state. Consider trying another node, or :meth:`disconnect` the player.


        .. versionadded:: 3.5.0
        """
        assert self._guild

        if new_node.identifier == self.node.identifier:
            msg: str = f"Player '{self._guild.id}' current node is identical to the passed node: {new_node!r}"
            raise InvalidNodeException(msg)

        await self._destroy(with_invalidate=False)
        self._node = new_node

        await self._dispatch_voice_update()
        if not self.connected:
            raise RuntimeError(f"Switching Node on player '{self._guild.id}' failed. Failed to switch voice_state.")

        self.node._players[self._guild.id] = self

        if not self._current:
            await self.set_filters(self.filters)
            await self.set_volume(self.volume)
            await self.pause(self.paused)
            return

        await self.play(
            self._current,
            replace=True,
            start=self.position,
            volume=self.volume,
            filters=self.filters,
            paused=self.paused,
        )
        logger.debug("Switching nodes for player: '%s' was successful. New Node: %r", self._guild.id, self.node)

    @property
    def inactive_channel_tokens(self) -> int | None:
        """A settable property which returns the token limit as an ``int`` of the amount of tracks to play before firing
        the :func:`on_wavelink_inactive_player` event when a channel is inactive.

        This property could return ``None`` if the check has been disabled.

        A channel is considered inactive when no real members (Members other than bots) are in the connected voice
        channel. On each consecutive track played without a real member in the channel, this token bucket will reduce
        by ``1``. After hitting ``0``, the :func:`on_wavelink_inactive_player` event will be fired and the token bucket
        will reset to the set value. The default value for this property is ``3``.

        This property can be set with any valid ``int`` or ``None``. If this property is set to ``<= 0`` or ``None``,
        the check will be disabled.

        Setting this property to ``1`` will fire the :func:`on_wavelink_inactive_player` event at the end of every track
        if no real members are in the channel and you have not disconnected the player.

        If this check successfully fires the :func:`on_wavelink_inactive_player` event, it will cancel any waiting
        :attr:`inactive_timeout` checks until a new track is played.

        The default for every player can be set on :class:`~wavelink.Node`.

        - See: :class:`~wavelink.Node`
        - See: :func:`on_wavelink_inactive_player`

        .. warning::

            Setting this property will reset the bucket.

        .. versionadded:: 3.4.0
        """
        return self._inactive_channel_limit

    @inactive_channel_tokens.setter
    def inactive_channel_tokens(self, value: int | None) -> None:
        if not value or value <= 0:
            self._inactive_channel_limit = None
            return

        self._inactive_channel_limit = value
        self._inactive_channel_count = value

    @property
    def inactive_timeout(self) -> int | None:
        """A property which returns the time as an ``int`` of seconds to wait before this player dispatches the
        :func:`on_wavelink_inactive_player` event.

        This property could return ``None`` if no time has been set.

        An inactive player is a player that has not been playing anything for the specified amount of seconds.

        - Pausing the player while a song is playing will not activate this countdown.
        - The countdown starts when a track ends and cancels when a track starts.
        - The countdown will not trigger until a song is played for the first time or this property is reset.
        - The default countdown for all players is set on :class:`~wavelink.Node`.

        This property can be set with a valid ``int`` of seconds to wait before dispatching the
        :func:`on_wavelink_inactive_player` event or ``None`` to remove the timeout.


        .. warning::

            Setting this to a value of ``0`` or below is the equivalent of setting this property to ``None``.


        When this property is set, the timeout will reset, and all previously waiting countdowns are cancelled.

        - See: :class:`~wavelink.Node`
        - See: :func:`on_wavelink_inactive_player`


        .. versionadded:: 3.2.0
        """
        return self._inactivity_wait

    @inactive_timeout.setter
    def inactive_timeout(self, value: int | None) -> None:
        if not value or value <= 0:
            self._inactivity_wait = None
            self._inactivity_cancel()
            return

        if value < 10:
            logger.warning('Setting "inactive_timeout" below 10 seconds may result in unwanted side effects.')

        self._inactivity_wait = value
        self._inactivity_cancel()

        if self.connected and not self.playing:
            self._inactivity_start()

    @property
    def autoplay(self) -> AutoPlayMode:
        """A property which returns the :class:`wavelink.AutoPlayMode` the player is currently in.

        This property can be set with any :class:`wavelink.AutoPlayMode` enum value.


        .. versionchanged:: 3.0.0

            This property now accepts and returns a :class:`wavelink.AutoPlayMode` enum value.
        """
        return self._autoplay

    @autoplay.setter
    def autoplay(self, value: Any) -> None:
        if not isinstance(value, AutoPlayMode):
            raise ValueError("Please provide a valid 'wavelink.AutoPlayMode' to set.")

        self._autoplay = value

    @property
    def node(self) -> Node:
        """The :class:`Player`'s currently selected :class:`Node`.


        .. versionchanged:: 3.0.0

            This property was previously known as ``current_node``.
        """
        return self._node

    @property
    def guild(self) -> discord.Guild | None:
        """Returns the :class:`Player`'s associated :class:`discord.Guild`.

        Could be None if this :class:`Player` has not been connected.
        """
        return self._guild

    @property
    def connected(self) -> bool:
        """Returns a bool indicating if the player is currently connected to a voice channel.

        .. versionchanged:: 3.0.0

            This property was previously known as ``is_connected``.
        """
        return self.channel and self._connected

    @property
    def current(self) -> Playable | None:
        """Returns the currently playing :class:`~wavelink.Playable` or None if no track is playing."""
        return self._current

    @property
    def volume(self) -> int:
        """Returns an int representing the currently set volume, as a percentage.

        See: :meth:`set_volume` for setting the volume.
        """
        return self._volume

    @property
    def filters(self) -> Filters:
        """Property which returns the :class:`~wavelink.Filters` currently assigned to the Player.

        See: :meth:`~wavelink.Player.set_filters` for setting the players filters.

        .. versionchanged:: 3.0.0

            This property was previously known as ``filter``.
        """
        return self._filters

    @property
    def paused(self) -> bool:
        """Returns the paused status of the player. A currently paused player will return ``True``.

        See: :meth:`pause` and :meth:`play` for setting the paused status.
        """
        return self._paused

    @property
    def ping(self) -> int:
        """Returns the ping in milliseconds as int between your connected Lavalink Node and Discord (Players Channel).

        Returns ``-1`` if no player update event has been received or the player is not connected.
        """
        return self._ping

    @property
    def playing(self) -> bool:
        """Returns whether the :class:`~Player` is currently playing a track and is connected.

        Due to relying on validation from Lavalink, this property may in some cases return ``True`` directly after
        skipping/stopping a track, although this is not the case when disconnecting the player.

        This property will return ``True`` in cases where the player is paused *and* has a track loaded.

        .. versionchanged:: 3.0.0

            This property used to be known as the `is_playing()` method.
        """
        return self._connected and self._current is not None

    @property
    def position(self) -> int:
        """Returns the position of the currently playing :class:`~wavelink.Playable` in milliseconds.

        This property relies on information updates from Lavalink.

        In cases there is no :class:`~wavelink.Playable` loaded or the player is not connected,
        this property will return ``0``.

        This property will return ``0`` if no update has been received from Lavalink.

        .. versionchanged:: 3.0.0

            This property now uses a monotonic clock.
        """
        if self.current is None or not self.playing:
            return 0

        if not self.connected:
            return 0

        if self._last_update is None:
            return 0

        if self.paused:
            return self._last_position

        position: int = int((time.monotonic_ns() - self._last_update) / 1000000) + self._last_position
        return min(position, self.current.length)

    def _update_event(self, payload: PlayerUpdateEventPayload) -> None:
        # Convert nanoseconds into milliseconds...
        self._last_update = time.monotonic_ns()
        self._last_position = payload.position

        self._ping = payload.ping

    async def on_voice_state_update(self, data: GuildVoiceStatePayload, /) -> None:
        channel_id = data["channel_id"]

        if not channel_id:
            await self._destroy()
            return

        self._connected = True

        self._voice_state["voice"]["session_id"] = data["session_id"]
        self.channel = self.client.get_channel(int(channel_id))  # type: ignore

    async def on_voice_server_update(self, data: VoiceServerUpdatePayload, /) -> None:
        self._voice_state["voice"]["token"] = data["token"]
        self._voice_state["voice"]["endpoint"] = data["endpoint"]

        await self._dispatch_voice_update()

    async def _dispatch_voice_update(self) -> None:
        assert self.guild is not None
        data: VoiceState = self._voice_state["voice"]

        session_id: str | None = data.get("session_id", None)
        token: str | None = data.get("token", None)
        endpoint: str | None = data.get("endpoint", None)

        if not session_id or not token or not endpoint:
            return

        request: RequestPayload = {"voice": {"sessionId": session_id, "token": token, "endpoint": endpoint, "channelId": str(self.channel.id) if self.channel else None}}

        try:
            await self.node._update_player(self.guild.id, data=request)
        except LavalinkException:
            await self.disconnect()
        else:
            self._connected = True
            self._connection_event.set()

        logger.debug("Player %s is dispatching VOICE_UPDATE.", self.guild.id)

    async def connect(
        self, *, timeout: float = 10.0, reconnect: bool, self_deaf: bool = False, self_mute: bool = False
    ) -> None:
        """

        .. warning::

            Do not use this method directly on the player. See: :meth:`discord.VoiceChannel.connect` for more details.


        Pass the :class:`wavelink.Player` to ``cls=`` in :meth:`discord.VoiceChannel.connect`.


        Raises
        ------
        ChannelTimeoutException
            Connecting to the voice channel timed out.
        InvalidChannelStateException
            You tried to connect this player without an appropriate voice channel.
        """
        if self.channel is MISSING:
            msg: str = 'Please use "discord.VoiceChannel.connect(cls=...)" and pass this Player to cls.'
            raise InvalidChannelStateException(f"Player tried to connect without a valid channel: {msg}")

        if not self._guild:
            self._guild = self.channel.guild

        self.node._players[self._guild.id] = self

        assert self.guild is not None
        await self.guild.change_voice_state(channel=self.channel, self_mute=self_mute, self_deaf=self_deaf)

        try:
            await asyncio.wait_for(self._connection_event.wait(), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            msg = f"Unable to connect to {self.channel} as it exceeded the timeout of {timeout} seconds."
            raise ChannelTimeoutException(msg)

    async def move_to(
        self,
        channel: VocalGuildChannel | None,
        *,
        timeout: float = 10.0,
        self_deaf: bool | None = None,
        self_mute: bool | None = None,
    ) -> None:
        """Method to move the player to another channel.

        Parameters
        ----------
        channel: :class:`discord.VoiceChannel` | :class:`discord.StageChannel`
            The new channel to move to.
        timeout: float
            The timeout in ``seconds`` before raising. Defaults to 10.0.
        self_deaf: bool | None
            Whether to deafen when moving. Defaults to ``None`` which keeps the current setting or ``False``
            if they can not be determined.
        self_mute: bool | None
            Whether to self mute when moving. Defaults to ``None`` which keeps the current setting or ``False``
            if they can not be determined.

        Raises
        ------
        ChannelTimeoutException
            Connecting to the voice channel timed out.
        InvalidChannelStateException
            You tried to connect this player without an appropriate guild.
        """
        if not self.guild:
            raise InvalidChannelStateException("Player tried to move without a valid guild.")

        self._connection_event.clear()
        self._reconnecting.clear()
        voice: discord.VoiceState | None = self.guild.me.voice

        if self_deaf is None and voice:
            self_deaf = voice.self_deaf

        if self_mute is None and voice:
            self_mute = voice.self_mute

        self_deaf = bool(self_deaf)
        self_mute = bool(self_mute)

        await self.guild.change_voice_state(channel=channel, self_mute=self_mute, self_deaf=self_deaf)

        if channel is None:
            self._reconnecting.set()
            return

        try:
            await asyncio.wait_for(self._connection_event.wait(), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            msg = f"Unable to connect to {channel} as it exceeded the timeout of {timeout} seconds."
            raise ChannelTimeoutException(msg)
        finally:
            self._reconnecting.set()

    async def play(
        self,
        track: Playable,
        *,
        replace: bool = True,
        start: int = 0,
        end: int | None = None,
        volume: int | None = None,
        paused: bool | None = None,
        add_history: bool = True,
        filters: Filters | None = None,
        populate: bool = False,
        max_populate: int = 5,
    ) -> Playable:
        """Play the provided :class:`~wavelink.Playable`.

        Parameters
        ----------
        track: :class:`~wavelink.Playable`
            The track to being playing.
        replace: bool
            Whether this track should replace the currently playing track, if there is one. Defaults to ``True``.
        start: int
            The position to start playing the track at in milliseconds.
            Defaults to ``0`` which will start the track from the beginning.
        end: Optional[int]
            The position to end the track at in milliseconds.
            Defaults to ``None`` which means this track will play until the very end.
        volume: Optional[int]
            Sets the volume of the player. Must be between ``0`` and ``1000``.
            Defaults to ``None`` which will not change the current volume.
            See Also: :meth:`set_volume`
        paused: bool | None
            Whether the player should be paused, resumed or retain current status when playing this track.
            Setting this parameter to ``True`` will pause the player. Setting this parameter to ``False`` will
            resume the player if it is currently paused. Setting this parameter to ``None`` will not change the status
            of the player. Defaults to ``None``.
        add_history: Optional[bool]
            If this argument is set to ``True``, the :class:`~Player` will add this track into the
            :class:`wavelink.Queue` history, if loading the track was successful. If ``False`` this track will not be
            added to your history. This does not directly affect the ``AutoPlay Queue`` but will alter how ``AutoPlay``
            recommends songs in the future. Defaults to ``True``.
        filters: Optional[:class:`~wavelink.Filters`]
            An Optional[:class:`~wavelink.Filters`] to apply when playing this track. Defaults to ``None``.
            If this is ``None`` the currently set filters on the player will be applied.
        populate: bool
            Whether the player should find and fill AutoQueue with recommended tracks based on the track provided.
            Defaults to ``False``.

            Populate will only search for recommended tracks when the current tracks has been accepted by Lavalink.
            E.g. if this method does not raise an error.

            You should consider when you use the ``populate`` keyword argument as populating the AutoQueue on every
            request could potentially lead to a large amount of tracks being populated.
        max_populate: int
            The maximum amount of tracks that should be added to the AutoQueue when the ``populate`` keyword argument is
            set to ``True``. This is NOT the exact amount of tracks that will be added. You should set this to a lower
            amount to avoid the AutoQueue from being overfilled.

            This argument has no effect when ``populate`` is set to ``False``.

            Defaults to ``5``.


        Returns
        -------
        :class:`~wavelink.Playable`
            The track that began playing.


        .. versionchanged:: 3.0.0

            Added the ``paused`` parameter. Parameters ``replace``, ``start``, ``end``, ``volume`` and ``paused``
            are now all keyword-only arguments.

            Added the ``add_history`` keyword-only argument.

            Added the ``filters`` keyword-only argument.


        .. versionchanged:: 3.3.0

            Added the ``populate`` keyword-only argument.
        """
        assert self.guild is not None

        original_vol: int = self._volume
        vol: int = volume or self._volume

        if vol != self._volume:
            self._volume = vol

        if replace or not self._current:
            self._current = track
            self._original = track

        old_previous = self._previous
        self._previous = self._current
        self.queue._loaded = track

        pause: bool = paused if paused is not None else self._paused

        if filters:
            self._filters = filters

        request: RequestPayload = {
            "track": {"encoded": track.encoded, "userData": track._user_data()},
            "volume": vol,
            "position": start,
            "endTime": end,
            "paused": pause,
            "filters": self._filters(),
        }

        try:
            await self.node._update_player(self.guild.id, data=request, replace=replace)
        except LavalinkException as e:
            self.queue._loaded = old_previous
            self._current = None
            self._original = None
            self._previous = old_previous
            self._volume = original_vol
            raise e

        self._paused = pause

        if add_history:
            assert self.queue.history is not None
            self.queue.history.put(track)

        if populate:
            await self._do_recommendation(populate_track=track, max_population=max_populate)

        return track

    async def pause(self, value: bool, /) -> None:
        """Set the paused or resume state of the player.

        Parameters
        ----------
        value: bool
            A bool indicating whether the player should be paused or resumed. True indicates that the player should be
            ``paused``. False will resume the player if it is currently paused.


        .. versionchanged:: 3.0.0

            This method now expects a positional-only bool value. The ``resume`` method has been removed.
        """
        assert self.guild is not None

        request: RequestPayload = {"paused": value}
        await self.node._update_player(self.guild.id, data=request)

        self._paused = value

    async def seek(self, position: int = 0, /) -> None:
        """Seek to the provided position in the currently playing track, in milliseconds.

        Parameters
        ----------
        position: int
            The position to seek to in milliseconds. To restart the song from the beginning,
            you can disregard this parameter or set position to 0.


        .. versionchanged:: 3.0.0

            The ``position`` parameter is now positional-only, and has a default of 0.
        """
        assert self.guild is not None

        if not self._current:
            return

        request: RequestPayload = {"position": position}
        await self.node._update_player(self.guild.id, data=request)

    async def set_filters(self, filters: Filters | None = None, /, *, seek: bool = False) -> None:
        """Set the :class:`wavelink.Filters` on the player.

        Parameters
        ----------
        filters: Optional[:class:`~wavelink.Filters`]
            The filters to set on the player. Could be ``None`` to reset the currently applied filters.
            Defaults to ``None``.
        seek: bool
            Whether to seek immediately when applying these filters. Seeking uses more resources, but applies the
            filters immediately. Defaults to ``False``.


        .. versionchanged:: 3.0.0

            This method now accepts a positional-only argument of filters, which now defaults to None. Filters
            were redesigned in this version, see: :class:`wavelink.Filters`.


        .. versionchanged:: 3.0.0

            This method was previously known as ``set_filter``.
        """
        assert self.guild is not None

        if filters is None:
            filters = Filters()

        request: RequestPayload = {"filters": filters()}
        await self.node._update_player(self.guild.id, data=request)
        self._filters = filters

        if self.playing and seek:
            await self.seek(self.position)

    async def set_volume(self, value: int = 100, /) -> None:
        """Set the :class:`Player` volume, as a percentage, between 0 and 1000.

        By default, every player is set to 100 on creation. If a value outside 0 to 1000 is provided it will be
        clamped.

        Parameters
        ----------
        value: int
            A volume value between 0 and 1000. To reset the player to 100, you can disregard this parameter.


        .. versionchanged:: 3.0.0

            The ``value`` parameter is now positional-only, and has a default of 100.
        """
        assert self.guild is not None
        vol: int = max(min(value, 1000), 0)

        request: RequestPayload = {"volume": vol}
        await self.node._update_player(self.guild.id, data=request)

        self._volume = vol

    async def disconnect(self, **kwargs: Any) -> None:
        """Disconnect the player from the current voice channel and remove it from the :class:`~wavelink.Node`.

        This method will cause any playing track to stop and potentially trigger the following events:

            - ``on_wavelink_track_end``
            - ``on_wavelink_websocket_closed``


        .. warning::

            Please do not re-use a :class:`Player` instance that has been disconnected, unwanted side effects are
            possible.
        """
        assert self.guild

        await self._destroy()
        await self.guild.change_voice_state(channel=None)

    async def stop(self, *, force: bool = True) -> Playable | None:
        """An alias to :meth:`skip`.

        See Also: :meth:`skip` for more information.

        .. versionchanged:: 3.0.0

            This method is now known as ``skip``, but the alias ``stop`` has been kept for backwards compatibility.
        """
        return await self.skip(force=force)

    async def skip(self, *, force: bool = True) -> Playable | None:
        """Stop playing the currently playing track.

        Parameters
        ----------
        force: bool
            Whether the track should skip looping, if :class:`wavelink.Queue` has been set to loop.
            Defaults to ``True``.

        Returns
        -------
        :class:`~wavelink.Playable` | None
            The currently playing track that was skipped, or ``None`` if no track was playing.


        .. versionchanged:: 3.0.0

            This method was previously known as ``stop``. To avoid confusion this method is now known as ``skip``.
            This method now returns the :class:`~wavelink.Playable` that was skipped.
        """
        assert self.guild is not None
        old: Playable | None = self._current

        if force:
            self.queue._loaded = None

        request: RequestPayload = {"track": {"encoded": None}}
        await self.node._update_player(self.guild.id, data=request, replace=True)

        return old

    def _invalidate(self) -> None:
        self._connected = False
        self._connection_event.clear()
        self._inactivity_cancel()

        task = self._recommend_task
        if task is not None and not task.done():
            task.cancel()
        self._recommend_task = None

        try:
            self.cleanup()
        except (AttributeError, KeyError):
            pass

    async def _destroy(self, with_invalidate: bool = True) -> None:
        assert self.guild

        if with_invalidate:
            self._invalidate()

        player: Player | None = self.node._players.pop(self.guild.id, None)

        if player:
            try:
                await self.node._destroy_player(self.guild.id)
            except Exception as e:
                logger.debug("Disregarding. Failed to send 'destroy_player' payload to Lavalink: %s", e)

    def _add_to_previous_seeds(self, seed: str) -> None:
        # deque(maxlen=...) drops the oldest seed when the cutoff is reached.
        self._previous_seeds.append(seed)
