---
sd_hide_title: true
---

# skretrieval

:::{grid-item}
:columns: 8
:class: sd-fs-3

The USask Atmospheric Retrieval Framework

:::

::::

`skretrieval` is an atmospheric retrieval (inverse problem) framework developed at the University of Saskatchewan. It provides
general utility methods for working with atmospheric retrieval problems.
`skretrieval` also provides generic, highly extendable, methods to retrieve atmospheric parameters from remote sensing measurements.

`skretrieval` is still in active development, and things are subject to change quickly, however it is already usable for many applications.
After [Installing](_installation) the package, we recommend starting with the [Quick Start Guide](_quickstart) to get a feel for the core concepts of the interface.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`paper-airplane;1.5em;sd-mr-1` Installation
:link: _installation
:link-type: ref

`skretrieval` is available both as a `conda` and `pip` package on most platforms.

+++
[Learn more »](installation)
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Quick Start Guide
:link: _quickstart
:link-type: ref

The quick start guide will help you set up your first retrieval simulation

+++
[Learn more »](quickstart)
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` User's Guide
:link: _users_guide
:link-type: ref

The user's guide demonstrates `skretrieval`'s features through example

+++
[Learn more »](users_guide)
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API Reference
:link: _api_reference
:link-type: ref

A full reference to `skretrieval`'s API.  This section assumes you are already
familiar with the core concepts of the model.

+++
[Learn more »](api_reference)
:::


::::

## License
`skretrieval` is made available under the MIT license (see [License](https://github.com/usask-arg/sasktran2/blob/main/license.md))


```{toctree}
:maxdepth: 4
:hidden:
:caption: For Users

installation
quickstart
users_guide
api_reference
changelog
```
