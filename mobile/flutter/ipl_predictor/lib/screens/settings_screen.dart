import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import '../theme/app_theme.dart';
import '../services/api_service.dart';
import '../widgets/glass_card.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _urlController = TextEditingController(text: ApiService.baseUrl);

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Settings', style: Theme.of(context).textTheme.headlineLarge),
            const SizedBox(height: 24),

            // API Configuration
            GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(children: [
                    const Icon(Icons.cloud_outlined, color: AppTheme.accent),
                    const SizedBox(width: 10),
                    Text('API Server', style: Theme.of(context).textTheme.titleLarge),
                  ]),
                  const SizedBox(height: 12),
                  TextField(
                    controller: _urlController,
                    style: const TextStyle(color: AppTheme.textPrimary),
                    decoration: InputDecoration(
                      hintText: 'https://your-api.up.railway.app',
                      hintStyle: const TextStyle(color: AppTheme.textSecondary),
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(10),
                        borderSide: const BorderSide(color: Color(0xFF2A2A4A)),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(10),
                        borderSide: const BorderSide(color: AppTheme.accent),
                      ),
                      filled: true,
                      fillColor: AppTheme.surface,
                    ),
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: () {
                        setState(() { ApiService.baseUrl = _urlController.text.trim(); });
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text('API URL updated'), backgroundColor: AppTheme.success),
                        );
                      },
                      style: ElevatedButton.styleFrom(backgroundColor: AppTheme.accent),
                      child: const Text('Save'),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 16),

            // About
            GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(children: [
                    const Icon(Icons.info_outline, color: AppTheme.accent),
                    const SizedBox(width: 10),
                    Text('About', style: Theme.of(context).textTheme.titleLarge),
                  ]),
                  const SizedBox(height: 12),
                  _aboutRow('Version', '1.0.0'),
                  _aboutRow('Model', 'XGBoost + LightGBM Ensemble'),
                  _aboutRow('Data Source', 'Cricsheet.org (ODC-BY)'),
                  _aboutRow('Framework', 'Flutter + FastAPI'),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: OutlinedButton.icon(
                      onPressed: () => launchUrl(Uri.parse('https://github.com/saisgadi-hash/ipl-prediction-agent')),
                      icon: const Icon(Icons.code),
                      label: const Text('View Source on GitHub'),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: AppTheme.textPrimary,
                        side: const BorderSide(color: Color(0xFF2A2A4A)),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 16),

            // Open Web Dashboard
            GlassCard(
              child: ListTile(
                contentPadding: EdgeInsets.zero,
                leading: const Icon(Icons.open_in_browser, color: AppTheme.accentPink),
                title: Text('Open Web Dashboard', style: Theme.of(context).textTheme.titleMedium),
                subtitle: Text('ipl-ai-prediction.up.railway.app', style: Theme.of(context).textTheme.bodySmall),
                trailing: const Icon(Icons.chevron_right, color: AppTheme.textSecondary),
                onTap: () => launchUrl(Uri.parse('https://ipl-ai-prediction.up.railway.app')),
              ),
            ),

            const SizedBox(height: 40),
            Center(
              child: Text('Built by Goutham with AI', style: Theme.of(context).textTheme.bodySmall),
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  Widget _aboutRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: Theme.of(context).textTheme.bodyMedium),
          Text(value, style: Theme.of(context).textTheme.bodyLarge),
        ],
      ),
    );
  }
}
