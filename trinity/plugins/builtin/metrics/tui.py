import asyncio
from typing import Dict, Iterable
import urwid

PALETTE = [
    ('body', 'black', 'light gray'),
    ('focus', 'light gray', 'dark blue', 'standout'),
    ('head', 'yellow', 'black', 'standout'),
    ('foot', 'light gray', 'black'),
    ('key', 'light cyan', 'black','underline'),
    ('title', 'white', 'black', 'bold'),
    ('flag', 'dark gray', 'light gray'),
    ('error', 'dark red', 'light gray'),
]


class ExampleTreeWidget(urwid.TreeWidget):
    """ Display widget for leaf nodes """
    def __init__(self, node, name):
        self.name = name
        super().__init__(node)

    def get_display_text(self):
        value = self.get_node().get_value()
        if isinstance(value, dict):
            return self.name
        return f'{self.name}' + " " + f'{value}'

    def keypress(self, size, key):
        """Handle expand & collapse requests (non-leaf nodes)"""
        # based on https://github.com/urwid/urwid/blob/master/urwid/treetools.py#L141
        if self.is_leaf:
            return key

        if key is " ":
            self.expanded = not self.expanded
            self.update_expanded_icon()
            return None

        return super().keypress(size, key)


class ExampleNode(urwid.TreeNode):
    """ Data storage object for leaf nodes """
    def __init__(self, name, data, parent, key, depth):
        self.name = name
        super().__init__(data, parent, key, depth)

    def load_widget(self):
        return ExampleTreeWidget(self, self.name)


class ExampleParentNode(urwid.ParentNode):
    """ Data storage object for interior/parent nodes """
    def __init__(self, name, data, parent=None, key=0, depth=0):
        self.name = name
        super().__init__(data, parent, key, depth)

    def load_widget(self):
        return ExampleTreeWidget(self, self.name)

    def load_child_keys(self):
        data = self.get_value()
        return range(len(data))

    def load_child_node(self, key):
        """Return either an ExampleNode or ExampleParentNode"""
        items = list(self.get_value().items())
        name, childdata = items[key]

        childdepth = self.get_depth() + 1
        if isinstance(childdata, dict):
            childclass = ExampleParentNode
        else:
            childclass = ExampleNode
        return childclass(name, childdata, parent=self, key=key, depth=childdepth)


example_tree = {
    'bonded_nodes': 7,
    'pool': {
        'handshake_attempts': 11,
        'concurrent_connection_attempts': 8,
    },
    'peers': {
        'received_msgs': 1,
    },
}



def set_in(subtree: Dict, keys: Iterable[str], value: int):
    assert len(keys) > 0

    key, *key_rest = keys

    if len(keys) == 1:
        subtree[key] = value
        return

    if key not in subtree:
        subtree[key] = {}

    set_in(subtree[key], key_rest, value)


def tree_from_stats(stats):
    retval = {}

    for key, value in stats.items():
        keys = key.split('.')
        set_in(retval, keys, value)

    return retval


class UrwidInterface:
    def __init__(self):
        self.urwid_loop = None
        self.should_exit = asyncio.Event()
        self.tree_walker = None

        self.build_view()

    def set_footer_text(self, text: str) -> None:
        self.footer_text.set_text(text)

    def set_data(self, stats: Dict[str, int]) -> None:
        tree = tree_from_stats(stats)
        parent_node = ExampleParentNode("metrics", tree)
        self.tree_walker.focus = parent_node
        self.listbox._invalidate()

        if self.urwid_loop.screen._started:
            self.urwid_loop.draw_screen()

    def build_view(self):
        ex = example_tree
        epn = ExampleParentNode("metrics", ex)
        self.tree_walker = urwid.TreeWalker(epn)
        listbox = urwid.TreeListBox(self.tree_walker)
        listbox.offset_rows = 1

        self.listbox = listbox

        self.footer_text = urwid.Text('not connected')

        header = urwid.AttrWrap(urwid.Text('Trinity Metrics'), "head")
        footer = urwid.AttrWrap(self.footer_text, "foot")

        self.view = urwid.Frame(
            listbox,
            header=header,
            footer=footer,
        )

    def exit_on_q(self, key):
        if key in ('q', 'Q'):
            self.should_exit.set()

            # TODO: figure out how to make this work
            # raise urwid.ExitMainLoop()

    def stop(self):
        self.should_exit.set()
        if self.urwid_loop:
            self.urwid_loop.screen.stop()

    async def run(self) -> None:
        loop = asyncio.get_event_loop()
        urwid_loop = urwid.MainLoop(
            self.view,
            palette=PALETTE,
            event_loop=urwid.AsyncioEventLoop(loop=loop),
            unhandled_input=self.exit_on_q,
        )
        self.urwid_loop = urwid_loop
        urwid_loop.start()

        await self.should_exit.wait()
