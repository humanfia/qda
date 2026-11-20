"""Register the commands, checks and suites this package adds to QDA."""


def register() -> None:
    """Add lazy cross-package registrations before the first registry read."""
    from qda_kl.eval.rundir import RUN_REPORTS
    from qda_kl.eval.suites import SUITES
    from qda_quant.cli import COMMANDS
    from qda_quant.doctor import CHECKS

    from qda_agent.components import COMPONENTS

    COMMANDS.register("probe", "qda_agent.probe:add_parser")
    CHECKS.register("disk room", "qda_agent.doctor:disk_room")
    CHECKS.register("search", "qda_agent.doctor:search")
    SUITES.register("livecodebench", "qda_agent.livecodebench:LiveCodeBenchSuite")
    RUN_REPORTS.register("qda-agent", "qda_agent.evaluate:report")
    COMPONENTS.register("fisher", "qda_agent.components:Fisher")
    COMPONENTS.register("probe", "qda_agent.components:Probe")
