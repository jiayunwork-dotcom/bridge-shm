import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from src.models.anomaly_alert import (
    AlertRule, AnomalyAlertEvent, RuleCondition, LogicOperator,
    MetricType, ComparisonType, PriorityLevel, AlertStatus,
    add_alert_event, load_alert_rules, update_alert_rule,
    generate_event_id
)
from src.models.unarchived_file import UnarchivedFile
from src.models.bridge import Bridge


DEFAULT_EVAL_INTERVAL = 10
DEFAULT_COOLDOWN_MINUTES = 5


class AnomalyDetector:
    def __init__(self, cooldown_minutes: int = DEFAULT_COOLDOWN_MINUTES):
        self.cooldown_minutes = cooldown_minutes
        self._duration_counters: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._condition_counters: Dict[str, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self._last_trigger_time: Dict[str, datetime] = {}

    @staticmethod
    def compute_rms(data: np.ndarray) -> float:
        if len(data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(data ** 2)))

    @staticmethod
    def compute_peak_to_peak(data: np.ndarray) -> float:
        if len(data) == 0:
            return 0.0
        return float(np.max(data) - np.min(data))

    @staticmethod
    def compute_fft(
        data: np.ndarray, sampling_rate: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(data) < 2:
            return np.array([0.0]), np.array([0.0])
        n = len(data)
        fft_vals = np.fft.rfft(data)
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
        fft_magnitude = np.abs(fft_vals) / n * 2
        return fft_freqs, fft_magnitude

    @staticmethod
    def compute_dominant_frequency(
        data: np.ndarray, sampling_rate: float
    ) -> float:
        freqs, mags = AnomalyDetector.compute_fft(data, sampling_rate)
        if len(mags) < 2:
            return 0.0
        valid_idx = freqs > 0.0
        if not np.any(valid_idx):
            return 0.0
        freqs = freqs[valid_idx]
        mags = mags[valid_idx]
        return float(freqs[np.argmax(mags)])

    @staticmethod
    def compute_baseline_drift(data: np.ndarray, baseline: Optional[float] = None) -> float:
        if len(data) == 0:
            return 0.0
        current_mean = float(np.mean(data))
        if baseline is None:
            return current_mean
        return current_mean - baseline

    def compute_metric(
        self,
        metric_type: MetricType,
        data: np.ndarray,
        sampling_rate: float,
        baseline_freq: Optional[float] = None,
        baseline_value: Optional[float] = None
    ) -> float:
        if metric_type == MetricType.RMS_AMPLITUDE:
            return self.compute_rms(data)
        elif metric_type == MetricType.PEAK_TO_PEAK:
            return self.compute_peak_to_peak(data)
        elif metric_type == MetricType.FREQUENCY_OFFSET:
            dominant = self.compute_dominant_frequency(data, sampling_rate)
            if baseline_freq is None or baseline_freq == 0:
                return 0.0
            return dominant - baseline_freq
        elif metric_type == MetricType.BASELINE_DRIFT:
            return self.compute_baseline_drift(data, baseline_value)
        return 0.0

    @staticmethod
    def check_threshold(
        metric_value: float,
        comparison: ComparisonType,
        threshold: float,
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None
    ) -> bool:
        if comparison == ComparisonType.GREATER_THAN:
            return metric_value > threshold
        elif comparison == ComparisonType.LESS_THAN:
            return metric_value < threshold
        elif comparison == ComparisonType.OUT_OF_RANGE:
            t_min = threshold_min if threshold_min is not None else threshold
            t_max = threshold_max if threshold_max is not None else threshold
            return metric_value < t_min or metric_value > t_max
        return False

    def is_in_cooldown(self, rule_id: str) -> bool:
        last = self._last_trigger_time.get(rule_id)
        if last is None:
            return False
        return datetime.now() - last < timedelta(minutes=self.cooldown_minutes)

    def _get_latest_unarchived_data(
        self, bridge_id: str, channel: int
    ) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str], Optional[float]]:
        files = UnarchivedFile.list_by_bridge(bridge_id)
        if not files:
            return None, None, None, None

        for uf in files:
            if uf.duration < 1.0:
                continue
            channel_name = f"CH_{channel:02d}"
            if channel_name not in uf.channel_names:
                continue
            df = uf.load_data()
            if df is None or channel_name not in df.columns:
                continue
            data = df[channel_name].values
            if len(data) > 0:
                return data, uf.sampling_rate, uf.id, uf.duration

        uf = files[0]
        if not uf.channel_names:
            return None, None, None, None
        ch_name = uf.channel_names[min(channel - 1, len(uf.channel_names) - 1)] if channel > 0 else uf.channel_names[0]
        df = uf.load_data()
        if df is None or ch_name not in df.columns:
            return None, None, None, None
        return df[ch_name].values, uf.sampling_rate, uf.id, uf.duration

    def _evaluate_single_condition(
        self,
        rule_id: str,
        condition_id: str,
        condition: RuleCondition,
        bridge: Bridge,
        eval_interval: int,
        baseline_freq: Optional[float] = None,
        baseline_value: Optional[float] = None
    ) -> Dict[int, Dict]:
        results: Dict[int, Dict] = {}

        for channel in condition.sensor_channels:
            data, sampling_rate, file_id, duration = self._get_latest_unarchived_data(bridge.id, channel)
            if data is None or sampling_rate is None or len(data) < sampling_rate:
                results[channel] = {"has_data": False}
                continue

            n_eval_samples = min(len(data), int(sampling_rate * eval_interval))
            eval_data = data[-n_eval_samples:]

            metric_value = self.compute_metric(
                condition.metric_type, eval_data, sampling_rate, baseline_freq, baseline_value
            )

            triggered = self.check_threshold(
                metric_value, condition.comparison, condition.threshold,
                condition.threshold_min, condition.threshold_max
            )

            counter_key = f"{rule_id}_{condition_id}"

            if triggered:
                self._condition_counters[rule_id][condition_id][channel] += eval_interval
                accumulated = self._condition_counters[rule_id][condition_id][channel]
                duration_met = accumulated >= condition.duration_seconds
            else:
                self._condition_counters[rule_id][condition_id][channel] = 0.0
                accumulated = 0.0
                duration_met = False

            results[channel] = {
                "has_data": True,
                "metric_value": metric_value,
                "triggered": triggered,
                "duration_met": duration_met,
                "accumulated": accumulated,
                "file_id": file_id,
                "duration": duration,
                "n_eval_samples": n_eval_samples,
                "sampling_rate": sampling_rate
            }

        return results

    def evaluate_rule(
        self,
        rule: AlertRule,
        bridge: Bridge,
        eval_interval: int = DEFAULT_EVAL_INTERVAL
    ) -> List[AnomalyAlertEvent]:
        new_events: List[AnomalyAlertEvent] = []

        if not rule.enabled:
            return new_events

        if self.is_in_cooldown(rule.id):
            return new_events

        if rule.is_composite and rule.conditions:
            new_events = self._evaluate_composite_rule(rule, bridge, eval_interval)
        else:
            new_events = self._evaluate_simple_rule(rule, bridge, eval_interval)

        if new_events:
            rules = load_alert_rules(bridge.id)
            for r in rules:
                if r.id == rule.id:
                    r.last_triggered = datetime.now()
                    break
            from src.models.anomaly_alert import save_alert_rules
            save_alert_rules(bridge.id, rules)

        return new_events

    def _evaluate_simple_rule(
        self,
        rule: AlertRule,
        bridge: Bridge,
        eval_interval: int
    ) -> List[AnomalyAlertEvent]:
        new_events: List[AnomalyAlertEvent] = []

        baseline_freq = None
        baseline_value = None
        if rule.metric_type in (MetricType.FREQUENCY_OFFSET, MetricType.BASELINE_DRIFT):
            first_ch = rule.sensor_channels[0] if rule.sensor_channels else 1
            data, sr, _, _ = self._get_latest_unarchived_data(bridge.id, first_ch)
            if data is not None and len(data) > 0:
                if rule.metric_type == MetricType.FREQUENCY_OFFSET:
                    baseline_freq = self.compute_dominant_frequency(data[:min(len(data), int(sr * 2))], sr)
                else:
                    baseline_value = float(np.mean(data[:min(len(data), int(sr * 2))]))

        for channel in rule.sensor_channels:
            data, sampling_rate, file_id, duration = self._get_latest_unarchived_data(bridge.id, channel)
            if data is None or sampling_rate is None or len(data) < sampling_rate:
                continue

            n_eval_samples = min(len(data), int(sampling_rate * eval_interval))
            eval_data = data[-n_eval_samples:]

            metric_value = self.compute_metric(
                rule.metric_type, eval_data, sampling_rate, baseline_freq, baseline_value
            )

            triggered = self.check_threshold(
                metric_value, rule.comparison, rule.threshold,
                rule.threshold_min, rule.threshold_max
            )

            if triggered:
                self._duration_counters[rule.id][channel] += eval_interval
                accumulated = self._duration_counters[rule.id][channel]

                if accumulated >= rule.duration_seconds:
                    offset = duration - (n_eval_samples / sampling_rate) if duration else 0.0
                    event = AnomalyAlertEvent(
                        id=generate_event_id(),
                        bridge_id=bridge.id,
                        rule_id=rule.id,
                        rule_name=rule.name,
                        trigger_time=datetime.now(),
                        sensor_channel=channel,
                        metric_value=round(metric_value, 6),
                        metric_type=rule.metric_type,
                        priority=rule.priority,
                        status=AlertStatus.PENDING,
                        unarchived_file_id=file_id,
                        trigger_offset_seconds=max(0.0, offset)
                    )
                    new_events.append(event)
                    self._duration_counters[rule.id][channel] = 0.0
                    self._last_trigger_time[rule.id] = datetime.now()
            else:
                self._duration_counters[rule.id][channel] = 0.0

        return new_events

    def _evaluate_composite_rule(
        self,
        rule: AlertRule,
        bridge: Bridge,
        eval_interval: int
    ) -> List[AnomalyAlertEvent]:
        new_events: List[AnomalyAlertEvent] = []
        is_and = rule.logic_operator == LogicOperator.AND

        all_condition_results: Dict[str, Dict[int, Dict]] = {}
        all_channels = set()

        for condition in rule.conditions:
            baseline_freq = None
            baseline_value = None
            if condition.metric_type in (MetricType.FREQUENCY_OFFSET, MetricType.BASELINE_DRIFT):
                first_ch = condition.sensor_channels[0] if condition.sensor_channels else 1
                data, sr, _, _ = self._get_latest_unarchived_data(bridge.id, first_ch)
                if data is not None and len(data) > 0:
                    if condition.metric_type == MetricType.FREQUENCY_OFFSET:
                        baseline_freq = self.compute_dominant_frequency(data[:min(len(data), int(sr * 2))], sr)
                    else:
                        baseline_value = float(np.mean(data[:min(len(data), int(sr * 2))]))

            results = self._evaluate_single_condition(
                rule.id, condition.id, condition, bridge, eval_interval,
                baseline_freq, baseline_value
            )
            all_condition_results[condition.id] = results
            for ch in results.keys():
                all_channels.add(ch)

        for channel in all_channels:
            skipped_conditions = []
            triggered_conditions = []
            duration_met_conditions = []
            has_data_conditions = []
            best_metric_value = 0.0
            best_file_id = None
            best_duration = 0.0
            best_n_eval_samples = 0
            best_sampling_rate = None

            for cond_idx, condition in enumerate(rule.conditions):
                cond_result = all_condition_results.get(condition.id, {})
                ch_result = cond_result.get(channel)

                if ch_result is None or not ch_result.get("has_data", False):
                    skipped_conditions.append(condition.id)
                    if is_and:
                        continue
                    else:
                        continue

                has_data_conditions.append(condition.id)
                if best_file_id is None:
                    best_file_id = ch_result.get("file_id")
                    best_duration = ch_result.get("duration", 0.0) or 0.0
                    best_n_eval_samples = ch_result.get("n_eval_samples", 0) or 0
                    best_sampling_rate = ch_result.get("sampling_rate")
                best_metric_value = max(best_metric_value, abs(ch_result.get("metric_value", 0.0)))

                if ch_result.get("triggered", False):
                    triggered_conditions.append(condition.id)
                if ch_result.get("duration_met", False):
                    duration_met_conditions.append(condition.id)

            should_fire = False
            if is_and:
                valid_conditions = [c for c in rule.conditions if c.id in has_data_conditions]
                if len(valid_conditions) > 0:
                    all_duration_met = all(c.id in duration_met_conditions for c in valid_conditions)
                    if all_duration_met:
                        should_fire = True
            else:
                if any(c.id in duration_met_conditions for c in rule.conditions):
                    should_fire = True

            if should_fire:
                offset = best_duration - (best_n_eval_samples / best_sampling_rate) if (best_duration and best_sampling_rate) else 0.0
                event = AnomalyAlertEvent(
                    id=generate_event_id(),
                    bridge_id=bridge.id,
                    rule_id=rule.id,
                    rule_name=rule.name,
                    trigger_time=datetime.now(),
                    sensor_channel=channel,
                    metric_value=round(best_metric_value, 6),
                    metric_type=rule.metric_type,
                    priority=rule.priority,
                    status=AlertStatus.PENDING,
                    unarchived_file_id=best_file_id,
                    trigger_offset_seconds=max(0.0, offset),
                    is_composite=True,
                    skipped_condition_ids=skipped_conditions,
                    triggered_condition_ids=triggered_conditions
                )
                new_events.append(event)
                self._last_trigger_time[rule.id] = datetime.now()
                for condition in rule.conditions:
                    if condition.id in duration_met_conditions:
                        self._condition_counters[rule.id][condition.id][channel] = 0.0

        return new_events

    def evaluate_all_rules(
        self,
        bridge: Bridge,
        eval_interval: int = DEFAULT_EVAL_INTERVAL
    ) -> List[AnomalyAlertEvent]:
        rules = load_alert_rules(bridge.id)
        all_new_events: List[AnomalyAlertEvent] = []

        for rule in rules:
            events = self.evaluate_rule(rule, bridge, eval_interval)
            for e in events:
                add_alert_event(bridge.id, e)
                all_new_events.append(e)

        return all_new_events


def get_waveform_around_trigger(
    bridge_id: str,
    file_id: str,
    channel: int,
    trigger_offset: float,
    seconds_before: float = 5.0,
    seconds_after: float = 5.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    uf = UnarchivedFile.load(bridge_id, file_id)
    if uf is None:
        return None, None, None

    channel_name = f"CH_{channel:02d}"
    if channel_name not in uf.channel_names:
        if not uf.channel_names:
            return None, None, None
        channel_name = uf.channel_names[0]

    df = uf.load_data()
    if df is None or channel_name not in df.columns:
        return None, None, None

    sr = uf.sampling_rate
    data = df[channel_name].values
    total_samples = len(data)
    trigger_sample = int(trigger_offset * sr)
    trigger_sample = max(0, min(trigger_sample, total_samples - 1))

    start_sample = max(0, trigger_sample - int(seconds_before * sr))
    end_sample = min(total_samples, trigger_sample + int(seconds_after * sr))

    clip_data = data[start_sample:end_sample]
    time_axis = np.arange(len(clip_data)) / sr - (trigger_sample - start_sample) / sr

    return time_axis, clip_data, sr


def get_spectrum_around_trigger(
    bridge_id: str,
    file_id: str,
    channel: int,
    trigger_offset: float,
    seconds_before: float = 5.0,
    seconds_after: float = 5.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    _, clip_data, sr = get_waveform_around_trigger(
        bridge_id, file_id, channel, trigger_offset, seconds_before, seconds_after
    )
    if clip_data is None or sr is None:
        return None, None
    return AnomalyDetector.compute_fft(clip_data, sr)
