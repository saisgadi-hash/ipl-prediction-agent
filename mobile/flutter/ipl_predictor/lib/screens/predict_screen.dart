import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../services/api_service.dart';
import '../models/prediction.dart';
import '../widgets/glass_card.dart';

class PredictScreen extends StatefulWidget {
  const PredictScreen({super.key});

  @override
  State<PredictScreen> createState() => _PredictScreenState();
}

class _PredictScreenState extends State<PredictScreen> {
  List<String> _teams = [];
  String? _team1;
  String? _team2;
  MatchPrediction? _prediction;
  bool _loading = false;
  bool _teamsLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadTeams();
  }

  Future<void> _loadTeams() async {
    final teams = await ApiService.getTeams();
    setState(() {
      _teams = teams;
      _team1 = teams.isNotEmpty ? teams[0] : null;
      _team2 = teams.length > 1 ? teams[5] : null;  // Default to MI
      _teamsLoading = false;
    });
  }

  Future<void> _predict() async {
    if (_team1 == null || _team2 == null || _team1 == _team2) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select two different teams'), backgroundColor: AppTheme.error),
      );
      return;
    }
    setState(() { _loading = true; _error = null; _prediction = null; });
    try {
      final result = await ApiService.predictMatch(team1: _team1!, team2: _team2!);
      setState(() { _prediction = result; _loading = false; });
    } catch (e) {
      setState(() { _error = e.toString(); _loading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Match Predictor', style: Theme.of(context).textTheme.headlineLarge),
            const SizedBox(height: 4),
            Text('Select two teams to predict the winner', style: Theme.of(context).textTheme.bodyMedium),
            const SizedBox(height: 24),

            // Team selectors
            if (_teamsLoading)
              const Center(child: CircularProgressIndicator(color: AppTheme.accent))
            else ...[
              _buildTeamSelector('Team 1', _team1, (val) => setState(() => _team1 = val)),
              const SizedBox(height: 12),
              // VS divider
              Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  decoration: BoxDecoration(
                    color: AppTheme.card,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: const Text('VS', style: TextStyle(color: AppTheme.accent, fontWeight: FontWeight.w700, fontSize: 16)),
                ),
              ),
              const SizedBox(height: 12),
              _buildTeamSelector('Team 2', _team2, (val) => setState(() => _team2 = val)),
              const SizedBox(height: 24),

              // Predict button
              SizedBox(
                width: double.infinity,
                height: 54,
                child: Container(
                  decoration: BoxDecoration(
                    gradient: AppTheme.accentGradient,
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: ElevatedButton(
                    onPressed: _loading ? null : _predict,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.transparent,
                      shadowColor: Colors.transparent,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                    ),
                    child: _loading
                        ? const SizedBox(width: 22, height: 22, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2.5))
                        : const Text('Predict Winner', style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600)),
                  ),
                ),
              ),
            ],

            // Error
            if (_error != null) ...[
              const SizedBox(height: 16),
              GlassCard(
                child: Row(children: [
                  const Icon(Icons.error_outline, color: AppTheme.error),
                  const SizedBox(width: 12),
                  Expanded(child: Text(_error!, style: const TextStyle(color: AppTheme.error))),
                ]),
              ),
            ],

            // Prediction result
            if (_prediction != null) ...[
              const SizedBox(height: 24),
              _buildResultCard(),
              if (_prediction!.topFactors.isNotEmpty) ...[
                const SizedBox(height: 16),
                _buildFactorsCard(),
              ],
              if (_prediction!.justification != null) ...[
                const SizedBox(height: 16),
                _buildJustificationCard(),
              ],
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildTeamSelector(String label, String? value, ValueChanged<String?> onChanged) {
    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: Theme.of(context).textTheme.bodySmall),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(
            value: value,
            onChanged: onChanged,
            isExpanded: true,
            dropdownColor: AppTheme.card,
            decoration: InputDecoration(
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: const BorderSide(color: AppTheme.surface)),
              enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: const BorderSide(color: AppTheme.border)),
              focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: const BorderSide(color: AppTheme.accent)),
              filled: true,
              fillColor: AppTheme.surface,
              contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            ),
            items: _teams.map((team) => DropdownMenuItem(
              value: team,
              child: Row(children: [
                Container(width: 10, height: 10, decoration: BoxDecoration(color: AppTheme.getTeamColor(team), shape: BoxShape.circle)),
                const SizedBox(width: 10),
                Flexible(child: Text(team, overflow: TextOverflow.ellipsis)),
              ]),
            )).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildResultCard() {
    final p = _prediction!;
    final winnerColor = AppTheme.getTeamColor(p.predictedWinner);
    final pct = (p.winProbability * 100).toStringAsFixed(1);
    final conf = (p.confidence * 100).toStringAsFixed(0);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [winnerColor.withOpacity(0.15), AppTheme.card],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: winnerColor.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          const Icon(Icons.emoji_events, color: AppTheme.warning, size: 36),
          const SizedBox(height: 10),
          Text('Predicted Winner', style: Theme.of(context).textTheme.bodyMedium),
          const SizedBox(height: 4),
          Text(p.predictedWinner, style: Theme.of(context).textTheme.headlineMedium?.copyWith(color: winnerColor)),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _statBubble('Win Prob', '$pct%', winnerColor),
              _statBubble('Confidence', '$conf%', AppTheme.accent),
            ],
          ),
        ],
      ),
    );
  }

  Widget _statBubble(String label, String value, Color color) {
    return Column(children: [
      Text(value, style: TextStyle(color: color, fontSize: 22, fontWeight: FontWeight.w700)),
      const SizedBox(height: 2),
      Text(label, style: Theme.of(context).textTheme.bodySmall),
    ]);
  }

  Widget _buildFactorsCard() {
    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Key Reasons', style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 12),
          ..._prediction!.topFactors.take(5).map((f) {
            final isPositive = f.direction == 'positive' || f.impact >= 0;
            return Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(isPositive ? Icons.trending_up : Icons.trending_down,
                    color: isPositive ? AppTheme.success : AppTheme.error, size: 20),
                  const SizedBox(width: 10),
                  Expanded(child: Text(f.description, style: Theme.of(context).textTheme.bodyLarge)),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _buildJustificationCard() {
    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            const Icon(Icons.auto_awesome, color: AppTheme.accent, size: 20),
            const SizedBox(width: 8),
            Text('AI Analysis', style: Theme.of(context).textTheme.titleLarge),
          ]),
          const SizedBox(height: 10),
          Text(_prediction!.justification!, style: Theme.of(context).textTheme.bodyLarge),
        ],
      ),
    );
  }
}
