/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * 
 * High-quality pre-loaded Knowledge Graph datasets for immediate play.
 */

import { Triple } from './kge';

export interface Dataset {
  id: string;
  name: string;
  description: string;
  triples: Triple[];
}

export const PRELOADED_DATASETS: Dataset[] = [
  {
    id: 'medical-discovery',
    name: 'Medical & Drug Discovery Pathway',
    description: 'A biological network connecting clinical symptoms, human diseases, cellular target proteins, and therapeutic inhibitors. Useful for discovering hidden drug-target candidates and off-target treatments.',
    triples: [
      // Diseases to Symptoms
      { id: 'm1', sub: 'Alzheimer_Disease', rel: 'presentsSymptom', obj: 'Memory_Loss' },
      { id: 'm2', sub: 'Alzheimer_Disease', rel: 'presentsSymptom', obj: 'Cognitive_Decline' },
      { id: 'm3', sub: 'Parkinson_Disease', rel: 'presentsSymptom', obj: 'Tremor' },
      { id: 'm4', sub: 'Parkinson_Disease', rel: 'presentsSymptom', obj: 'Motor_Rigidity' },
      { id: 'm5', sub: 'Rheumatoid_Arthritis', rel: 'presentsSymptom', obj: 'Joint_Inflammation' },
      { id: 'm6', sub: 'Rheumatoid_Arthritis', rel: 'presentsSymptom', obj: 'Chronic_Pain' },
      { id: 'm7', sub: 'Type_2_Diabetes', rel: 'presentsSymptom', obj: 'Insulin_Resistance' },
      { id: 'm8', sub: 'Type_2_Diabetes', rel: 'presentsSymptom', obj: 'Hyperglycemia' },

      // Diseases to Target Proteins
      { id: 'm9', sub: 'Alzheimer_Disease', rel: 'associatedWithProtein', obj: 'Amyloid_Beta_PP' },
      { id: 'm10', sub: 'Alzheimer_Disease', rel: 'associatedWithProtein', obj: 'Tau_Protein' },
      { id: 'm11', sub: 'Parkinson_Disease', rel: 'associatedWithProtein', obj: 'Alpha_Synuclein' },
      { id: 'm12', sub: 'Rheumatoid_Arthritis', rel: 'associatedWithProtein', obj: 'TNF_Alpha' },
      { id: 'm13', sub: 'Rheumatoid_Arthritis', rel: 'associatedWithProtein', obj: 'JAK_Kinase' },
      { id: 'm14', sub: 'Type_2_Diabetes', rel: 'associatedWithProtein', obj: 'AMPK_Enzyme' },
      { id: 'm15', sub: 'Type_2_Diabetes', rel: 'associatedWithProtein', obj: 'GLP1_Receptor' },

      // Proteins to Biological Mechanisms
      { id: 'm16', sub: 'Amyloid_Beta_PP', rel: 'triggersMechanism', obj: 'Plaque_Aggregation' },
      { id: 'm17', sub: 'Tau_Protein', rel: 'triggersMechanism', obj: 'Neurofibrillary_Tangles' },
      { id: 'm18', sub: 'Alpha_Synuclein', rel: 'triggersMechanism', obj: 'Lewy_Body_Formation' },
      { id: 'm19', sub: 'TNF_Alpha', rel: 'triggersMechanism', obj: 'Cytokine_Storm' },
      { id: 'm20', sub: 'JAK_Kinase', rel: 'triggersMechanism', obj: 'Immune_Signaling' },
      { id: 'm21', sub: 'AMPK_Enzyme', rel: 'regulatesMechanism', obj: 'Glucose_Uptake' },
      { id: 'm22', sub: 'GLP1_Receptor', rel: 'regulatesMechanism', obj: 'Insulin_Secretion' },

      // Drugs to Inhibited Proteins / Activated Proteins
      { id: 'm23', sub: 'Aducanumab_Drug', rel: 'inhibitsProtein', obj: 'Amyloid_Beta_PP' },
      { id: 'm24', sub: 'L_DOPA_Drug', rel: 'counteractsProtein', obj: 'Alpha_Synuclein' },
      { id: 'm25', sub: 'Adalimumab_Drug', rel: 'inhibitsProtein', obj: 'TNF_Alpha' },
      { id: 'm26', sub: 'Tofacitinib_Drug', rel: 'inhibitsProtein', obj: 'JAK_Kinase' },
      { id: 'm27', sub: 'Metformin_Drug', rel: 'activatesProtein', obj: 'AMPK_Enzyme' },
      { id: 'm28', sub: 'Semaglutide_Drug', rel: 'activatesProtein', obj: 'GLP1_Receptor' },

      // Drugs to Clinical Indication (Treatment) - SOME NOT CONFIRMED (the KGE should discover them!)
      { id: 'm29', sub: 'Aducanumab_Drug', rel: 'approvedForDisease', obj: 'Alzheimer_Disease' },
      { id: 'm30', sub: 'L_DOPA_Drug', rel: 'approvedForDisease', obj: 'Parkinson_Disease' },
      { id: 'm31', sub: 'Adalimumab_Drug', rel: 'approvedForDisease', obj: 'Rheumatoid_Arthritis' },
      { id: 'm32', sub: 'Semaglutide_Drug', rel: 'approvedForDisease', obj: 'Type_2_Diabetes' },
      
      // Additional connections for complex linkages (enabling latent path reasoning)
      { id: 'm33', sub: 'Joint_Inflammation', rel: 'modulatedBy', obj: 'TNF_Alpha' },
      { id: 'm34', sub: 'Insulin_Resistance', rel: 'modulatedBy', obj: 'AMPK_Enzyme' },
      { id: 'm35', sub: 'Cognitive_Decline', rel: 'associatedWith', obj: 'Plaque_Aggregation' },
    ]
  },
  {
    id: 'greek-pantheon',
    name: 'Greek Mythology Pantheon',
    description: 'A rich relational map of historical kinship, martial rivalries, domains, and epic weapons among ancient Greek deities. Ideal for tracking transitive family trees and domains.',
    triples: [
      // Parentage
      { id: 'g1', sub: 'Cronus', rel: 'parentOf', obj: 'Zeus' },
      { id: 'g2', sub: 'Cronus', rel: 'parentOf', obj: 'Poseidon' },
      { id: 'g3', sub: 'Cronus', rel: 'parentOf', obj: 'Hades' },
      { id: 'g4', sub: 'Cronus', rel: 'parentOf', obj: 'Hera' },
      { id: 'g5', sub: 'Rhea', rel: 'parentOf', obj: 'Zeus' },
      { id: 'g6', sub: 'Zeus', rel: 'parentOf', obj: 'Ares' },
      { id: 'g7', sub: 'Zeus', rel: 'parentOf', obj: 'Athena' },
      { id: 'g8', sub: 'Zeus', rel: 'parentOf', obj: 'Apollo' },
      { id: 'g9', sub: 'Zeus', rel: 'parentOf', obj: 'Artemis' },
      { id: 'g10', sub: 'Zeus', rel: 'parentOf', obj: 'Hermes' },
      { id: 'g11', sub: 'Hera', rel: 'parentOf', obj: 'Ares' },
      { id: 'g12', sub: 'Leto', rel: 'parentOf', obj: 'Apollo' },
      { id: 'g13', sub: 'Leto', rel: 'parentOf', obj: 'Artemis' },

      // Marriages & Alliances
      { id: 'g14', sub: 'Zeus', rel: 'marriedTo', obj: 'Hera' },
      { id: 'g15', sub: 'Hera', rel: 'marriedTo', obj: 'Zeus' },
      { id: 'g16', sub: 'Cronus', rel: 'marriedTo', obj: 'Rhea' },
      { id: 'g17', sub: 'Rhea', rel: 'marriedTo', obj: 'Cronus' },

      // Domains & Roles
      { id: 'g18', sub: 'Zeus', rel: 'rulesDomain', obj: 'The_Sky' },
      { id: 'g19', sub: 'Poseidon', rel: 'rulesDomain', obj: 'The_Oceans' },
      { id: 'g20', sub: 'Hades', rel: 'rulesDomain', obj: 'The_Underworld' },
      { id: 'g21', sub: 'Ares', rel: 'rulesDomain', obj: 'Warfare' },
      { id: 'g22', sub: 'Athena', rel: 'rulesDomain', obj: 'Wisdom_and_Strategy' },
      { id: 'g23', sub: 'Apollo', rel: 'rulesDomain', obj: 'Music_and_Prophecy' },
      { id: 'g24', sub: 'Artemis', rel: 'rulesDomain', obj: 'The_Wild_and_Hunt' },

      // Weapons & Symbols
      { id: 'g25', sub: 'Zeus', rel: 'wieldsWeapon', obj: 'Thunderbolt' },
      { id: 'g26', sub: 'Poseidon', rel: 'wieldsWeapon', obj: 'Trident' },
      { id: 'g27', sub: 'Hades', rel: 'wieldsWeapon', obj: 'Helm_of_Darkness' },
      { id: 'g28', sub: 'Ares', rel: 'wieldsWeapon', obj: 'Spear_and_Shield' },
      { id: 'g29', sub: 'Athena', rel: 'wieldsWeapon', obj: 'Aegis_Shield' },
      { id: 'g30', sub: 'Apollo', rel: 'wieldsWeapon', obj: 'Golden_Bow' },
      { id: 'g31', sub: 'Artemis', rel: 'wieldsWeapon', obj: 'Silver_Bow' },

      // Affiliations & Locations
      { id: 'g32', sub: 'Zeus', rel: 'residesIn', obj: 'Mount_Olympus' },
      { id: 'g33', sub: 'Hera', rel: 'residesIn', obj: 'Mount_Olympus' },
      { id: 'g34', sub: 'Poseidon', rel: 'residesIn', obj: 'Mount_Olympus' }, // Also ocean
      { id: 'g35', sub: 'Athena', rel: 'residesIn', obj: 'Mount_Olympus' },
      { id: 'g36', sub: 'Ares', rel: 'residesIn', obj: 'Mount_Olympus' },
      { id: 'g37', sub: 'Hades', rel: 'residesIn', obj: 'The_Underworld' },
    ]
  },
  {
    id: 'tech-giants',
    name: 'Tech Enterprise Strategy',
    description: 'An enterprise network capturing high-tech company products, structural subsidiaries, competitive targets, and cloud platforms. Useful for analyzing market dynamics and synergy vectors.',
    triples: [
      // Company and Subsidiaries/Acquisitions
      { id: 't1', sub: 'Google', rel: 'ownsSubsidiary', obj: 'YouTube' },
      { id: 't2', sub: 'Google', rel: 'ownsSubsidiary', obj: 'DeepMind' },
      { id: 't3', sub: 'Google', rel: 'ownsSubsidiary', obj: 'Waymo' },
      { id: 't4', sub: 'Meta', rel: 'ownsSubsidiary', obj: 'Instagram' },
      { id: 't5', sub: 'Meta', rel: 'ownsSubsidiary', obj: 'WhatsApp' },
      { id: 't6', sub: 'Microsoft', rel: 'ownsSubsidiary', obj: 'LinkedIn' },
      { id: 't7', sub: 'Microsoft', rel: 'ownsSubsidiary', obj: 'GitHub' },
      { id: 't8', sub: 'Microsoft', rel: 'ownsSubsidiary', obj: 'OpenAI_MajorStake' },
      { id: 't9', sub: 'Amazon', rel: 'ownsSubsidiary', obj: 'Twitch' },
      { id: 't10', sub: 'Amazon', rel: 'ownsSubsidiary', obj: 'Whole_Foods' },

      // Product Ecosystems
      { id: 't11', sub: 'Google', rel: 'developedProduct', obj: 'Android_OS' },
      { id: 't12', sub: 'Google', rel: 'developedProduct', obj: 'Google_Cloud_Platform' },
      { id: 't13', sub: 'Google', rel: 'developedProduct', obj: 'Gemini_AI' },
      { id: 't14', sub: 'Microsoft', rel: 'developedProduct', obj: 'Windows_OS' },
      { id: 't15', sub: 'Microsoft', rel: 'developedProduct', obj: 'Azure_Cloud' },
      { id: 't16', sub: 'Microsoft', rel: 'developedProduct', obj: 'Copilot_AI' },
      { id: 't17', sub: 'Amazon', rel: 'developedProduct', obj: 'Amazon_Web_Services' },
      { id: 't18', sub: 'Amazon', rel: 'developedProduct', obj: 'Alexa_Voice_Assistant' },
      { id: 't19', sub: 'Meta', rel: 'developedProduct', obj: 'Llama_AI_Model' },
      { id: 't20', sub: 'Meta', rel: 'developedProduct', obj: 'Quest_VR_Headset' },
      { id: 't21', sub: 'Apple', rel: 'developedProduct', obj: 'iOS_Mobile' },
      { id: 't22', sub: 'Apple', rel: 'developedProduct', obj: 'MacBook_Hardware' },
      { id: 't23', sub: 'Apple', rel: 'developedProduct', obj: 'Apple_Intelligence_AI' },

      // Core Sectors
      { id: 't24', sub: 'Google_Cloud_Platform', rel: 'competesInSector', obj: 'Cloud_Computing' },
      { id: 't25', sub: 'Azure_Cloud', rel: 'competesInSector', obj: 'Cloud_Computing' },
      { id: 't26', sub: 'Amazon_Web_Services', rel: 'competesInSector', obj: 'Cloud_Computing' },
      { id: 't27', sub: 'Gemini_AI', rel: 'competesInSector', obj: 'Generative_AI' },
      { id: 't28', sub: 'Copilot_AI', rel: 'competesInSector', obj: 'Generative_AI' },
      { id: 't29', sub: 'Llama_AI_Model', rel: 'competesInSector', obj: 'Generative_AI' },
      { id: 't30', sub: 'Apple_Intelligence_AI', rel: 'competesInSector', obj: 'Generative_AI' },
      { id: 't31', sub: 'Android_OS', rel: 'competesInSector', obj: 'Mobile_Operating_Systems' },
      { id: 't32', sub: 'iOS_Mobile', rel: 'competesInSector', obj: 'Mobile_Operating_Systems' },

      // Direct Competitions
      { id: 't33', sub: 'Google', rel: 'primaryCompetitor', obj: 'Microsoft' },
      { id: 't34', sub: 'Microsoft', rel: 'primaryCompetitor', obj: 'Google' },
      { id: 't35', sub: 'Meta', rel: 'primaryCompetitor', obj: 'ByteDance_TikTok' },
      { id: 't36', sub: 'Amazon_Web_Services', rel: 'directRival', obj: 'Azure_Cloud' },
      { id: 't37', sub: 'Google_Cloud_Platform', rel: 'directRival', obj: 'Azure_Cloud' },
      { id: 't38', sub: 'Gemini_AI', rel: 'directRival', obj: 'Copilot_AI' },
    ]
  }
];
